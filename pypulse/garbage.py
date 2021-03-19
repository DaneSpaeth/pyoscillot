import matplotlib.pyplot as plt
    save = False
    if save:
        star = GridStar(N_star=50, N_border=1, vsini=3000)
    else:
        cols = 5
        # fig, ax = plt.subplots(1, cols)
        fig, ax = plt.subplots((1, 3), figsize=(12, 9))
    # line = 6254.29
    # wave, spec = star.calc_spectrum(line, line)
    # spec = spec / np.max(spec)
    # bis_wave, bis = bisector_new(wave, spec)

    # center = wave[spec.argmin()]
    # plt.plot(bis_wave, bis, marker=".", linestyle="None")
    # plt.show()
    # exit()
    # N = 12
    # phases = np.linspace(0, 1 - (1 / N), N - 1)
    phases = np.array([0, 0.25, 0.5, 0.75, 1.0])
    phases = np.array([0.25, 0.5, 0.75])
    m = np.array([2, 4, 6])
    for idx, p in enumerate(phases):
        line = 6254.29
        if save:
            star.add_pulsation(l=4, m=-4, k=0.15, phase=p)

            wave, spec = star.calc_spectrum(line, line)
            plt.imshow(star.pulsation.real, origin="lower",
                       vmin=-200, vmax=200, cmap="seismic")
            plt.savefig(f"arrays/{round(p, 4)}.pdf")
            # plt.show()
            # exit()
            plt.close()
            if not idx:
                np.save("arrays/wave.npy", wave)
            np.save(f"arrays/{round(p,4)}.npy", spec)
            continue
        if not idx:
            wave = np.load("arrays/wave.npy")
            # wave = wave[::100]
        spec = np.load(f"arrays/{round(p,4)}.npy")
        # print(wave
        # spec = spec[::100]

        # print(len(wave))

        row = int(idx / cols)
        col = idx % cols
        center = wave[spec.argmin()]

        # print((wave[-1] - wave[0]) / len(wave) * C / 1e3)
        bis_wave, bis, center = bisector_new(wave, spec)
        vs = ((bis_wave - line) / line) * C
        vs = vs + 280
        # ax[col].plot(vs, bis)
        # # ax[row, col].set_xlim(-600, 600)
        # # ax[row, col].plot((wave - center) / center * C,
        # #                   spec, label=(center - 7000) / center * C)
        # ax[col].set_xlabel("V [m/s]")
        # ax[col].set_xlim(-120, 120)
        # # ax[row, col].set_xlim(np.min(bis_wave), np.max(bis_wave))
        # # ax[row, col].axvline(center)
        # ax[col].legend()

        ax.plot(vs, bis, label=p - 0.25)

    ax.legend()
    plt.xlabel("V_r (m/s)")
    plt.ylabel("Relative Flux")
    plt.title("Bisector for l=4, m=-4, pulsation phases")
    plt.show()

    # plt.show()
