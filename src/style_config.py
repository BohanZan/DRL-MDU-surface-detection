"""
Shared visual style configuration for all project figures.

TOL muted palette — colorblind-friendly, paper-ready.
"""
# MDU trajectory colors (4 agents: blue, gold, rose, olive)
TOL_MUTED = ["#004488", "#EECC66", "#994455", "#997700"]


def apply_style():
    """Apply Times New Roman + bold labels to current matplotlib context."""
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.weight"] = "normal"
