import re
import ast
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (1) CONFIG – modifiable facilement
# ============================================================
METHOD_A_NAME = "Random"
METHOD_B_NAME = "k-means+entropy"
METHOD_C_NAME = "entropy+k-means"  # <-- change le nom ici

# On ne génère PLUS que le graphique F1
OUT_F1 = "f1_comparison_3hybride.png"

# Champ à parser dans les logs
F1_KEY = "F1(spam)"

# Options de plot
ALPHA_BAND = 0.20
MARKER = "o"

# Baseline AVANT active learning -> iter 0
BASELINE_F1 = 0.2411
BASELINE_FRAC = 0.05  # (pas utilisé dans ce script mais gardé si tu veux annoter)


# ============================================================
# (2) COLLE TES LOGS ICI
# ============================================================
TEXT_METHOD_A = r"""
iter 0/5 | labeled_SMS_target=223/4459 (0.05)
  ACC mean=0.4978  CI95=[0.4671, 0.5284]  runs=['0.5004', '0.5085', '0.4843']
  PREC(spam) mean=0.1910 CI95=[0.1830, 0.1989] runs=['0.1946', '0.1897', '0.1886']
  REC(spam) mean=0.8523  CI95=[0.7797, 0.9250]  runs=['0.8725', '0.8188', '0.8658']
  F1(spam) mean=0.3120   CI95=[0.2985, 0.3256]  runs=['0.3182', '0.3081', '0.3097']

iter 1/5 | SMS_frac=0.2
  ACC mean=0.6888  CI95=[0.6398, 0.7378]   runs=['0.6691', '0.7085', '0.6888']
  PREC(spam) mean=0.2849 CI95=[0.2462, 0.3235] runs=['0.2689', '0.3000', '0.2857']
  REC(spam) mean=0.8770  CI95=[0.8385, 0.9155]   runs=['0.8591', '0.8859', '0.8859']
  F1(spam) mean=0.4300   CI95=[0.3818, 0.4781]   runs=['0.4096', '0.4482', '0.4321']

iter 2/5 | SMS_frac=0.4
  ACC mean=0.7925  CI95=[0.7815, 0.8035]   runs=['0.7874', '0.7955', '0.7946']
  PREC(spam) mean=0.3836 CI95=[0.3694, 0.3978] runs=['0.3771', '0.3862', '0.3876']
  REC(spam) mean=0.9105  CI95=[0.8758, 0.9452]   runs=['0.9060', '0.8993', '0.9262']
  F1(spam) mean=0.5398   CI95=[0.5224, 0.5572]   runs=['0.5325', '0.5403', '0.5465']

iter 3/5 | SMS_frac=0.6
  ACC mean=0.8236  CI95=[0.8000, 0.8472]   runs=['0.8251', '0.8135', '0.8323']
  PREC(spam) mean=0.4256 CI95=[0.3895, 0.4616] runs=['0.4286', '0.4098', '0.4383']
  REC(spam) mean=0.9105  CI95=[0.8758, 0.9452]   runs=['0.9262', '0.8993', '0.9060']
  F1(spam) mean=0.5799   CI95=[0.5431, 0.6168]   runs=['0.5860', '0.5630', '0.5908']

iter 4/5 | SMS_frac=0.8
  ACC mean=0.8529  CI95=[0.8245, 0.8814]   runs=['0.8556', '0.8628', '0.8404']
  PREC(spam) mean=0.4749 CI95=[0.4248, 0.5250] runs=['0.4789', '0.4928', '0.4531']
  REC(spam) mean=0.9239  CI95=[0.8892, 0.9586]   runs=['0.9128', '0.9195', '0.9396']
  F1(spam) mean=0.6271   CI95=[0.5893, 0.6648]   runs=['0.6282', '0.6417', '0.6114']

iter 5/5 | SMS_frac=1.0
  ACC mean=0.8849  CI95=[0.8647, 0.9051]   runs=['0.8816', '0.8789', '0.8942']
  PREC(spam) mean=0.5412 CI95=[0.4919, 0.5904] runs=['0.5331', '0.5267', '0.5638']
  REC(spam) mean=0.9217  CI95=[0.9121, 0.9313]   runs=['0.9195', '0.9262', '0.9195']
  F1(spam) mean=0.6818   CI95=[0.6446, 0.7190]   runs=['0.6749', '0.6715', '0.6990']
"""

TEXT_METHOD_B = r"""
iter 0/5 | labeled_SMS_target=223/4459 (0.05)
  ACC mean=0.4547  CI95=[0.4353, 0.4741]  runs=['0.4601', '0.4583', '0.4457']
  PREC(spam) mean=0.1837 CI95=[0.1706, 0.1968] runs=['0.1796', '0.1896', '0.1818']
  REC(spam) mean=0.8949  CI95=[0.7944, 0.9954]  runs=['0.8523', '0.9329', '0.8993']
  F1(spam) mean=0.3048   CI95=[0.2813, 0.3283]  runs=['0.2967', '0.3152', '0.3025']
  
iter 1/5
  ACC mean=0.6359 CI95=[0.5699,0.7018] runs=['0.6484', '0.6054', '0.6538']
  F1(spam) mean=0.3993 CI95=[0.3450,0.4536] runs=['0.4006', '0.3768', '0.4204']

iter 2/5
  ACC mean=0.7668 CI95=[0.7411,0.7925] runs=['0.7767', '0.7677', '0.7561']
  F1(spam) mean=0.5095 CI95=[0.5006,0.5184] runs=['0.5108', '0.5122', '0.5055']

iter 3/5
  ACC mean=0.8293 CI95=[0.8098,0.8488] runs=['0.8314', '0.8359', '0.8206']
  F1(spam) mean=0.5902 CI95=[0.5613,0.6192] runs=['0.5913', '0.6013', '0.5781']

iter 4/5
  ACC mean=0.8732 CI95=[0.8610,0.8855] runs=['0.8700', '0.8709', '0.8789']
  F1(spam) mean=0.6636 CI95=[0.6395,0.6876] runs=['0.6572', '0.6588', '0.6747']

iter 5/5
  ACC mean=0.8843 CI95=[0.8653,0.9033] runs=['0.8924', '0.8771', '0.8834']
  F1(spam) mean=0.6848 CI95=[0.6453,0.7242] runs=['0.7015', '0.6699', '0.6829']
"""

TEXT_METHOD_C = r"""
iter 0/5 | labeled_SMS_target=223/4459 (0.05)
  ACC mean=0.4120  CI95=[0.3836, 0.4403]  runs=['0.4063', '0.4251', '0.4045']
  PREC(spam) mean=0.1780 CI95=[0.1673, 0.1887] runs=['0.1757', '0.1830', '0.1753']
  REC(spam) mean=0.9396  CI95=[0.9107, 0.9685]  runs=['0.9329', '0.9530', '0.9329']
  F1(spam) mean=0.2993   CI95=[0.2826, 0.3159]  runs=['0.2957', '0.3070', '0.2951']
  
iter 1/5 | labeled_SMS_target=892/4459 (0.2)
  ACC mean=0.6502  CI95=[0.5875, 0.7130]  runs=['0.6789', '0.6404', '0.6314']
  PREC(spam) mean=0.2672 CI95=[0.2270, 0.3073] runs=['0.2854', '0.2614', '0.2547']
  REC(spam) mean=0.9239  CI95=[0.8985, 0.9494]  runs=['0.9329', '0.9262', '0.9128']
  F1(spam) mean=0.4143   CI95=[0.3640, 0.4647]  runs=['0.4371', '0.4077', '0.3982']

iter 2/5 | labeled_SMS_target=1784/4459 (0.4)
  ACC mean=0.7764  CI95=[0.7445, 0.8082]  runs=['0.7794', '0.7623', '0.7874']
  PREC(spam) mean=0.3658 CI95=[0.3299, 0.4018] runs=['0.3700', '0.3497', '0.3778']
  REC(spam) mean=0.9150  CI95=[0.8895, 0.9405]  runs=['0.9262', '0.9060', '0.9128']
  F1(spam) mean=0.5226   CI95=[0.4834, 0.5618]  runs=['0.5287', '0.5047', '0.5344']

iter 3/5 | labeled_SMS_target=2675/4459 (0.6)
  ACC mean=0.8281  CI95=[0.7947, 0.8615]  runs=['0.8413', '0.8143', '0.8287']
  PREC(spam) mean=0.4341 CI95=[0.3833, 0.4848] runs=['0.4545', '0.4137', '0.4340']
  REC(spam) mean=0.9329  CI95=[0.9162, 0.9496]  runs=['0.9396', '0.9329', '0.9262']
  F1(spam) mean=0.5923   CI95=[0.5432, 0.6414]  runs=['0.6127', '0.5732', '0.5910']

iter 4/5 | labeled_SMS_target=3567/4459 (0.8)
  ACC mean=0.8631  CI95=[0.8386, 0.8875]  runs=['0.8744', '0.8574', '0.8574']
  PREC(spam) mean=0.4941 CI95=[0.4448, 0.5433] runs=['0.5170', '0.4825', '0.4828']
  REC(spam) mean=0.9284  CI95=[0.9029, 0.9539]  runs=['0.9195', '0.9262', '0.9396']
  F1(spam) mean=0.6447   CI95=[0.6076, 0.6818]  runs=['0.6618', '0.6345', '0.6378']

iter 5/5 | labeled_SMS_target=4459/4459 (1.0)
  ACC mean=0.8849  CI95=[0.8647, 0.9051]  runs=['0.8816', '0.8789', '0.8942']
  PREC(spam) mean=0.5412 CI95=[0.4919, 0.5904] runs=['0.5331', '0.5267', '0.5638']
  REC(spam) mean=0.9217  CI95=[0.9121, 0.9313]  runs=['0.9195', '0.9262', '0.9195']
  F1(spam) mean=0.6818   CI95=[0.6446, 0.7190]  runs=['0.6749', '0.6715', '0.6990']
"""


# ============================================================
# (3) Parsing
# ============================================================
def _parse_runs_list(runs_str: str) -> np.ndarray:
    vals = ast.literal_eval(runs_str)  # parse sécurisé d'une liste python
    return np.array([float(v) for v in vals], dtype=float)


def parse_logs_f1_only(text: str, f1_key: str):
    """
    Retourne :
      iters,
      f1_mean, f1_min, f1_max
    """
    iter_headers = list(re.finditer(r"iter\s+(\d+)/5", text))
    if not iter_headers:
        raise ValueError("Aucun bloc 'iter X/5' trouvé.")

    iters = []
    f1_mean, f1_min, f1_max = [], [], []

    f1_re = re.compile(rf"{re.escape(f1_key)}\s+mean=([0-9.]+).*?runs=(\[[^\]]+\])")

    for idx, m in enumerate(iter_headers):
        it = int(m.group(1))
        start = m.start()
        end = iter_headers[idx + 1].start() if idx + 1 < len(iter_headers) else len(text)
        block = text[start:end]

        f1_m = f1_re.search(block)
        if not f1_m:
            raise ValueError(f"F1 manquant pour iter {it}")

        f1_runs = _parse_runs_list(f1_m.group(2))

        iters.append(it)
        f1_mean.append(float(f1_m.group(1)))
        f1_min.append(f1_runs.min())
        f1_max.append(f1_runs.max())

    order = np.argsort(iters)
    return (
        np.array(iters)[order],
        np.array(f1_mean)[order],
        np.array(f1_min)[order],
        np.array(f1_max)[order],
    )


def add_baseline_and_shift(iters, mean, vmin, vmax, baseline_value):
    """
    - Décale toutes les itérations existantes de +1 (iter 0 -> iter 1, etc.)
    - Ajoute iter 0 = baseline (avant active learning), avec min=max=mean=baseline_value
    """
    it_shift = iters + 1
    it_new = np.concatenate([np.array([0], dtype=int), it_shift])
    mean_new = np.concatenate([np.array([baseline_value], dtype=float), mean])
    vmin_new = np.concatenate([np.array([baseline_value], dtype=float), vmin])
    vmax_new = np.concatenate([np.array([baseline_value], dtype=float), vmax])
    return it_new, mean_new, vmin_new, vmax_new


# ============================================================
# (4) Plot
# ============================================================
def _plot_one(it, mean, vmin, vmax, name):
    plt.plot(it, mean, marker=MARKER, label=name)
    plt.fill_between(it, vmin, vmax, alpha=ALPHA_BAND)


def plot_three_methods_f1_with_band(
    itA, meanA, minA, maxA, nameA,
    itB, meanB, minB, maxB, nameB,
    itC, meanC, minC, maxC, nameC,
    title, ylabel, outfile
):
    plt.figure()

    # A
    _plot_one(itA, meanA, minA, maxA, nameA)

    # B (optionnel)
    if itB is not None:
        _plot_one(itB, meanB, minB, maxB, nameB)

    # C (optionnel)
    if itC is not None:
        _plot_one(itC, meanC, minC, maxC, nameC)

    xticks = sorted(
        set(
            itA.tolist()
            + (itB.tolist() if itB is not None else [])
            + (itC.tolist() if itC is not None else [])
        )
    )
    plt.xticks(xticks)

    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")


def main():
    # Parse méthode A
    itA, f1A, f1Amin, f1Amax = parse_logs_f1_only(TEXT_METHOD_A, F1_KEY)

    # Parse méthode B (si fournie)
    have_b = ("iter" in TEXT_METHOD_B) and (F1_KEY in TEXT_METHOD_B)
    if have_b:
        itB, f1B, f1Bmin, f1Bmax = parse_logs_f1_only(TEXT_METHOD_B, F1_KEY)
    else:
        itB = f1B = f1Bmin = f1Bmax = None

    # Parse méthode C (si fournie)
    have_c = ("iter" in TEXT_METHOD_C) and (F1_KEY in TEXT_METHOD_C)
    if have_c:
        itC, f1C, f1Cmin, f1Cmax = parse_logs_f1_only(TEXT_METHOD_C, F1_KEY)
    else:
        itC = f1C = f1Cmin = f1Cmax = None

    # Baseline + shift
    itA, f1A, f1Amin, f1Amax = add_baseline_and_shift(itA, f1A, f1Amin, f1Amax, BASELINE_F1)
    if have_b:
        itB, f1B, f1Bmin, f1Bmax = add_baseline_and_shift(itB, f1B, f1Bmin, f1Bmax, BASELINE_F1)
    if have_c:
        itC, f1C, f1Cmin, f1Cmax = add_baseline_and_shift(itC, f1C, f1Cmin, f1Cmax, BASELINE_F1)

    # Plot
    plot_three_methods_f1_with_band(
        itA, f1A, f1Amin, f1Amax, METHOD_A_NAME,
        itB, f1B, f1Bmin, f1Bmax, METHOD_B_NAME if have_b else None,
        itC, f1C, f1Cmin, f1Cmax, METHOD_C_NAME if have_c else None,
        title="F1(spam) vs Iterations",
        ylabel="F1(spam) (SMS test)",
        outfile=OUT_F1
    )

    print(f"Saved: {OUT_F1}")
    print("Done.")


if __name__ == "__main__":
    main()
