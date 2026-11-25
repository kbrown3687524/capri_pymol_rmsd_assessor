"""
capri_assess.py  --  CAPRI-style assessment of docked models against a reference.

    run capri_assess.py
    assess_all()

PyMOL is used only to read coordinates. Residue correspondence comes from
sequence alignment, so renumbered chains are handled; superposition is Kabsch
in numpy, so no cmd.align, no cmd.create, no PyMOL version quirks.

Metrics, all standard CAPRI:
  L-RMSD  receptor superposed onto the reference receptor, ligand then measured
          WHERE IT LANDS -- no second fit.
  I-RMSD  Kabsch RMSD over reference interface residues of both partners.
  Fnat    fraction of native residue-residue contacts recovered.

RMSD throughout is   sqrt( (1/N) * sum_i |r_i(model) - r_i(ref)|^2 ).
"""

from pymol import cmd
import numpy as np
import difflib

# ===========================================================================
# CONFIG
# ===========================================================================
REF            = "5JNX"
REF_RECEPTOR   = "E"     # VP24
REF_LIGAND     = "A"     # KPNA5 C-terminal domain

MODEL_RECEPTOR = None    # let it find VP24 by sequence
MODEL_LIGAND   = "H"     # GRAMM ligand chain

LIGAND_COPIES     = 1
LIGAND_NATIVE_LEN = None

CONTACT_CUT = 5.0
IFACE_CUT   = 10.0
MIN_SEQ_ID  = 0.30
# ===========================================================================

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}


# --------------------------------------------------------------------------
# reading structures.  NB: never use x, y or z as space-dict keys in
# cmd.iterate -- PyMOL reserves them and raises a NameError.
# --------------------------------------------------------------------------

ALT = "(alt '' or alt 'A')"   # keep one conformer only -- a duplicated CA
                              # silently corrupts the sequence/coordinate mapping


def ca_trace(obj, ch):
    """([resi...], sequence string, Nx3 CA coords) in file order, one altloc."""
    rows = []
    cmd.iterate_state(1, "%s and chain %s and polymer and name CA and %s"
                      % (obj, ch, ALT),
                      "rows.append((int(resi), resn, x, y, z))",
                      space={"rows": rows, "int": int})
    resis = [r[0] for r in rows]
    seq = "".join(AA3.get(r[1], "X") for r in rows)
    xyz = np.array([[r[2], r[3], r[4]] for r in rows], dtype=float)
    return resis, seq, xyz


def heavy_atoms(obj, ch):
    """([resi...], Nx3 coords) for all non-hydrogen polymer atoms."""
    rows = []
    cmd.iterate_state(1, "%s and chain %s and polymer and not hydro and %s"
                      % (obj, ch, ALT),
                      "rows.append((int(resi), x, y, z))",
                      space={"rows": rows, "int": int})
    return ([r[0] for r in rows],
            np.array([[r[1], r[2], r[3]] for r in rows], dtype=float))


def chains_of(obj):
    out = []
    for ch in cmd.get_chains(obj):
        resis, seq, xyz = ca_trace(obj, ch)
        if len(resis) != len(set(resis)):
            dup = sorted({r for r in resis if resis.count(r) > 1})
            print("   WARNING %s chain %s: duplicated residues in CA trace %s "
                  "-- superposition will be wrong" % (obj, ch, dup[:10]))
        if len(resis):
            out.append({"chain": ch, "resis": resis, "seq": seq, "xyz": xyz,
                        "n": len(resis), "lo": min(resis), "hi": max(resis)})
    return sorted(out, key=lambda d: -d["n"])


# --------------------------------------------------------------------------
# maths
# --------------------------------------------------------------------------

def kabsch(P, Q):
    """Rotation+translation mapping P onto Q, and the fitted RMSD."""
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    U, S, Vt = np.linalg.svd(P0.T.dot(Q0))
    d = np.sign(np.linalg.det(Vt.T.dot(U.T)))
    R = Vt.T.dot(np.diag([1.0, 1.0, d])).dot(U.T)
    rmsd = np.sqrt(((P0.dot(R.T) - Q0) ** 2).sum() / len(P))
    return R, Qc - R.dot(Pc), rmsd


def apply_rt(X, R, t):
    return X.dot(R.T) + t


def inplace_rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum() / len(A)))


def correspond(seq_a, resi_a, seq_b, resi_b):
    """Index pairs (i in a, j in b) from sequence alignment."""
    sm = difflib.SequenceMatcher(None, seq_a, seq_b, autojunk=False)
    pairs = []
    for i, j, n in sm.get_matching_blocks():
        pairs.extend((i + k, j + k) for k in range(n))
    return pairs


def seq_id(a, b):
    """Matched residues as a fraction of the SHORTER sequence.

    Not difflib's ratio(), which divides by the combined length and so scores
    a perfect 13-residue peptide against full-length RBBP6 at about 0.014.
    """
    if not a or not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(blk.size for blk in sm.get_matching_blocks())
    return matched / float(min(len(a), len(b)))


def contacts(resis_a, xyz_a, resis_b, xyz_b, cut):
    """Set of (resi_a, resi_b) pairs within cut, via a spatial hash."""
    grid = {}
    for idx, c in enumerate(xyz_b):
        grid.setdefault(tuple((c // cut).astype(int)), []).append(idx)
    c2, pairs = cut * cut, set()
    for ia, ca in enumerate(xyz_a):
        gx, gy, gz = (ca // cut).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for ib in grid.get((gx + dx, gy + dy, gz + dz), []):
                        if ((ca - xyz_b[ib]) ** 2).sum() <= c2:
                            pairs.add((resis_a[ia], resis_b[ib]))
    return pairs


def capri_class(fnat, L, I):
    if fnat >= 0.5 and (L <= 1.0 or I <= 1.0):
        return "High"
    if fnat >= 0.3 and (L <= 5.0 or I <= 2.0):
        return "Medium"
    if fnat >= 0.1 and (L <= 10.0 or I <= 4.0):
        return "Acceptable"
    return "Incorrect"


# --------------------------------------------------------------------------
# role assignment
# --------------------------------------------------------------------------

def reference_roles(ref=REF):
    chs = {c["chain"]: c for c in chains_of(ref)}
    order = chains_of(ref)
    rec = REF_RECEPTOR or order[0]["chain"]
    if REF_LIGAND:
        lig = REF_LIGAND
    else:
        ra, xa = heavy_atoms(ref, rec)
        best, bestn = None, -1
        for c in order:
            if c["chain"] == rec:
                continue
            rb, xb = heavy_atoms(ref, c["chain"])
            n = len(contacts(ra, xa, rb, xb, CONTACT_CUT))
            if n > bestn:
                best, bestn = c["chain"], n
        lig = best
    return chs[rec], chs[lig]


def match_chain(model_chains, target_seq, exclude=(), label=""):
    cands = [c for c in model_chains if c["chain"] not in exclude]
    if not cands:
        return None
    scored = sorted(((seq_id(c["seq"], target_seq), c) for c in cands),
                    key=lambda p: -p[0])
    best_score, best = scored[0]
    if best_score >= MIN_SEQ_ID:
        return best
    print("      %s: best chain %s scored %.3f (need %.2f); all chains: %s"
          % (label or "match", best["chain"], best_score, MIN_SEQ_ID,
             ", ".join("%s=%.2f/%dres" % (c["chain"], s, c["n"])
                       for s, c in scored)))
    return None


def ligand_blocks(lig):
    """[(lo_index, hi_index)] slices of the ligand chain, one per copy."""
    n = lig["n"]
    if LIGAND_COPIES <= 1:
        return [(0, n)]
    size = LIGAND_NATIVE_LEN or (n // LIGAND_COPIES)
    return [(s, min(s + size, n)) for s in range(0, n, size)]


# --------------------------------------------------------------------------
# assessment
# --------------------------------------------------------------------------

def assess_one(model, ref_rec, ref_lig, ref=REF, verbose=True):
    mchains = chains_of(model)
    m_rec = match_chain(mchains, ref_rec["seq"], label="%s receptor" % model)
    if m_rec is None:
        print("%-32s  no chain matches the reference receptor" % model)
        return None
    m_lig = match_chain(mchains, ref_lig["seq"], exclude=(m_rec["chain"],),
                        label="%s ligand" % model)
    if m_lig is None:
        print("%-32s  no chain matches the reference ligand" % model)
        return None

    # --- receptor superposition, from sequence correspondence --------------
    rp = correspond(m_rec["seq"], m_rec["resis"], ref_rec["seq"], ref_rec["resis"])
    if len(rp) < 3:
        print("%-32s  receptor correspondence too short (%d)" % (model, len(rp)))
        return None
    P = m_rec["xyz"][[i for i, j in rp]]
    Q = ref_rec["xyz"][[j for i, j in rp]]
    R, t, rec_fit = kabsch(P, Q)

    # --- ligand, measured where the receptor fit puts it -------------------
    best = None
    for lo, hi in ligand_blocks(m_lig):
        sub_seq = m_lig["seq"][lo:hi]
        sub_res = m_lig["resis"][lo:hi]
        sub_xyz = m_lig["xyz"][lo:hi]
        lp = correspond(sub_seq, sub_res, ref_lig["seq"], ref_lig["resis"])
        if len(lp) < 3:
            continue
        A = apply_rt(sub_xyz[[i for i, j in lp]], R, t)
        B = ref_lig["xyz"][[j for i, j in lp]]
        L = inplace_rmsd(A, B)
        if best is None or L < best["L_RMSD"]:
            best = {"L_RMSD": L, "n_lig": len(lp), "slice": (lo, hi), "lp": lp}
    if best is None:
        print("%-32s  no usable ligand correspondence" % model)
        return None

    lo, hi = best["slice"]

    # ligand atoms that correspond to the reference ligand, in the reference
    # frame and keyed by reference residue number. Used for the pairwise
    # comparison so every model is compared over the SAME residues as L-RMSD,
    # not over whole chains whose disordered regions would dominate.
    A_best = apply_rt(m_lig["xyz"][lo:hi][[i for i, j in best["lp"]]], R, t)
    lig_by_ref = {ref_lig["resis"][j]: A_best[k]
                  for k, (i, j) in enumerate(best["lp"])}

    # --- Fnat ---------------------------------------------------------------
    rr, rx = heavy_atoms(ref, ref_rec["chain"])
    lr, lx = heavy_atoms(ref, ref_lig["chain"])
    native = contacts(rr, rx, lr, lx, CONTACT_CUT)

    mrr, mrx = heavy_atoms(model, m_rec["chain"])
    mlr, mlx = heavy_atoms(model, m_lig["chain"])
    keep = [k for k, r in enumerate(mlr)
            if m_lig["resis"][lo] <= r <= m_lig["resis"][hi - 1]]
    mlr = [mlr[k] for k in keep]
    mlx = mlx[keep]
    pred_raw = contacts(mrr, apply_rt(mrx, R, t), mlr, apply_rt(mlx, R, t),
                        CONTACT_CUT)
    # translate model residue numbers onto reference numbering
    rec_map = {m_rec["resis"][i]: ref_rec["resis"][j] for i, j in rp}
    lig_map = {m_lig["resis"][lo + i]: ref_lig["resis"][j] for i, j in best["lp"]}
    pred = {(rec_map[a], lig_map[b]) for a, b in pred_raw
            if a in rec_map and b in lig_map}
    fnat = len(native & pred) / float(len(native)) if native else 0.0

    # --- I-RMSD -------------------------------------------------------------
    iface_rec = {a for a, b in contacts(rr, rx, lr, lx, IFACE_CUT)}
    iface_lig = {b for a, b in contacts(rr, rx, lr, lx, IFACE_CUT)}
    P_i, Q_i = [], []
    for i, j in rp:
        if ref_rec["resis"][j] in iface_rec:
            P_i.append(m_rec["xyz"][i]); Q_i.append(ref_rec["xyz"][j])
    for i, j in best["lp"]:
        if ref_lig["resis"][j] in iface_lig:
            P_i.append(m_lig["xyz"][lo + i]); Q_i.append(ref_lig["xyz"][j])
    if len(P_i) >= 3:
        _, _, irms = kabsch(np.array(P_i), np.array(Q_i))
    else:
        irms = float("nan")

    cls = capri_class(fnat, best["L_RMSD"], irms if irms == irms else 999)
    out = {"model": model, "L_RMSD": best["L_RMSD"], "I_RMSD": irms,
           "Fnat": fnat, "CAPRI": cls, "rec_fit": rec_fit,
           "n_rec": len(rp), "n_lig": best["n_lig"], "n_native": len(native),
           "m_rec": m_rec["chain"], "m_lig": m_lig["chain"],
           "R": R, "t": t, "slice": (lo, hi), "lig_obj": m_lig,
           "lig_by_ref": lig_by_ref}

    if verbose:
        print("%-32s %8.2f %8.2f %7.3f  %-11s  rec %s (%d at %.2f A), "
              "lig %s (%d)"
              % (model, out["L_RMSD"], irms, fnat, cls, m_rec["chain"],
                 len(rp), rec_fit, m_lig["chain"], best["n_lig"]))
    return out


def assess_all(ref=REF, out="capri_assessment.csv"):
    ref_rec, ref_lig = reference_roles(ref)
    rr, rx = heavy_atoms(ref, ref_rec["chain"])
    lr, lx = heavy_atoms(ref, ref_lig["chain"])
    ncon = len(contacts(rr, rx, lr, lx, CONTACT_CUT))

    print("\nReference %s" % ref)
    print("   receptor chain %s  (%d res, %d-%d)  -- superposed on"
          % (ref_rec["chain"], ref_rec["n"], ref_rec["lo"], ref_rec["hi"]))
    print("   ligand   chain %s  (%d res, %d-%d)  -- measured in place"
          % (ref_lig["chain"], ref_lig["n"], ref_lig["lo"], ref_lig["hi"]))
    print("   %d native residue contacts at %.1f A" % (ncon, CONTACT_CUT))
    if LIGAND_COPIES > 1:
        print("   ligand copies per model: %d" % LIGAND_COPIES)
    print("\n%-32s %8s %8s %7s  %-11s  %s"
          % ("model", "L-RMSD", "I-RMSD", "Fnat", "CAPRI", "chains used"))
    print("-" * 120)

    models = [o for o in cmd.get_object_list() if o != ref]
    res = [r for r in (assess_one(m, ref_rec, ref_lig, ref)
                       for m in sorted(models)) if r]

    with open(out, "w") as fh:
        fh.write("model,L_RMSD,I_RMSD,Fnat,CAPRI,receptor_fit_RMSD,"
                 "n_receptor_atoms,n_ligand_atoms,n_native_contacts\n")
        for r in res:
            fh.write("%s,%.3f,%.3f,%.3f,%s,%.3f,%d,%d,%d\n"
                     % (r["model"], r["L_RMSD"], r["I_RMSD"], r["Fnat"],
                        r["CAPRI"], r["rec_fit"], r["n_rec"], r["n_lig"],
                        r["n_native"]))
    print("\nWritten to %s" % out)

    print("\nPairwise ligand RMSD between models, over the SAME residues used")
    print("for L-RMSD (those present in the reference ligand), each model")
    print("placed by its own receptor fit. Small here alongside large L-RMSD")
    print("above means the tools agree on a wrong answer.")
    print("-" * 120)
    names = sorted(r["model"] for r in res)
    by_name = {r["model"]: r["lig_by_ref"] for r in res}
    vals = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(by_name[a]) & set(by_name[b]))
            if len(shared) < 3:
                continue
            A = np.array([by_name[a][k] for k in shared])
            B = np.array([by_name[b][k] for k in shared])
            v = inplace_rmsd(A, B)
            vals.append(v)
            print("%-32s %-32s %8.2f  (n=%d)" % (a, b, v, len(shared)))
    if vals:
        print("\nmin %.2f A, median %.2f A, max %.2f A over %d pairs"
              % (min(vals), float(np.median(vals)), max(vals), len(vals)))
    return res


cmd.extend("assess_all", assess_all)
cmd.extend("reference_roles", reference_roles)

# clear helper names left behind by earlier versions of this script, which
# otherwise surface as 'Invalid selection name "_tmp"'
for _stale in ("_tmp", "tmpcopy", "_ifk", "_ifv", "_ifr", "_ifres",
               "selIfRec", "selIfLig", "ref_face", "face_mod", "occluded"):
    try:
        cmd.delete(_stale)
    except Exception:
        pass

print("Loaded. REF = %s, LIGAND_COPIES = %d. Run: assess_all()"
      % (REF, LIGAND_COPIES))
