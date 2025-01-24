import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from flamo import dsp, system
from flamo.functional import db2mag, mag2db, get_magnitude, get_eigenvalues, WGN_reverb
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer

torch.manual_seed(130297)

# TODO: test the adapt code for GPU
# ==================================================================================================
# ===================================== Active Acoustics Class =====================================


class AA(nn.Module):
    """
    Active Acoustics (AA) model presented at DAFx24, 3-7 September, Guilford, UK.
    Reference:
        De Bortoli G., Dal Santo G., Prawda K., Lokki T., Välimäki V., and Schlecht S. J.
        Differentiable Active Acoustics---Optimizing Stability via Gradient Descent
        Int. Conf. on Digital Audio Effects (DAFx), Sep. 2024.
    """

    def __init__(
        self,
        n_S: int,
        n_M: int,
        n_L: int,
        n_A: int,
        fs: int = 48000,
        nfft: int = 2**11,
        FIR_order: int = 100,
        wgn_RT: float = 1.0,
        alias_decay_db: float = 0,
    ):
        r"""
        Initialize the Active Acoustics (AA) model.
        Stores system parameters, RIRs, and filters.

            **Args**:
                - n_S (int): number of natural sound sources.
                - n_M (int): number of microphones.
                - n_L (int): number of loudspeakers.
                - n_A (int): number of audience positions.
                - fs (int, optional): sampling frequency. Defaults to 48000.
                - nfft (int, optional): number of frequency bins. Defaults to 2**11.
                - FIR_order (int, optional): order of the FIR filters. Defaults to 100.
                - wgn_RT (float, optional): reverberation time of the WGN reverb. Defaults to 1.0.
                - alias_decay_db (float, optional): Time-alias decay in dB. Defaults to 0.
        """
        nn.Module.__init__(self)

        # Processing resolution
        self.fs = fs
        self.nfft = nfft

        # Sources, transducers, and audience
        self.n_S = n_S
        self.n_M = n_M
        self.n_L = n_L
        self.n_A = n_A

        # Physical room
        self.__Room = AA_RIRs(
            dir="./rirs/Otala-2024.05.10",
            n_S=self.n_S,
            n_L=self.n_L,
            n_M=self.n_M,
            n_A=self.n_A,
            fs=self.fs,
        )
        self.H_SM = dsp.Filter(
            size=(self.__Room.RIR_length, n_M, n_S),
            nfft=self.nfft,
            alias_decay_db=alias_decay_db,
        )
        self.H_SM.assign_value(self.__Room.get_scs_to_mcs())
        self.H_SA = dsp.Filter(
            size=(self.__Room.RIR_length, n_A, n_S),
            nfft=self.nfft,
            alias_decay_db=alias_decay_db,
        )
        self.H_SA.assign_value(self.__Room.get_scs_to_aud())
        self.H_LM = dsp.Filter(
            size=(self.__Room.RIR_length, n_M, n_L),
            nfft=self.nfft,
            alias_decay_db=alias_decay_db,
        )
        self.H_LM.assign_value(self.__Room.get_lds_to_mcs())
        self.H_LA = dsp.Filter(
            size=(self.__Room.RIR_length, n_A, n_L),
            nfft=self.nfft,
            alias_decay_db=alias_decay_db,
        )
        self.H_LA.assign_value(self.__Room.get_lds_to_aud())

        # Virtual room
        self.G = dsp.parallelGain(
            size=(self.n_L,), nfft=self.nfft, alias_decay_db=alias_decay_db
        )
        self.G.assign_value(torch.ones(self.n_L))
        fir_matrix = dsp.Filter(
            size=(FIR_order, self.n_L, self.n_M),
            nfft=self.nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
        )
        wgn_rev = WGN_reverb(matrix_size=(self.n_L,), t60=wgn_RT, samplerate=self.fs)
        wgn_matrix = dsp.parallelFilter(
            size=wgn_rev.shape, nfft=self.nfft, alias_decay_db=alias_decay_db
        )
        wgn_matrix.assign_value(wgn_rev)

        self.V_ML = OrderedDict([("U", fir_matrix), ("R", wgn_matrix)])

        # Feedback loop
        self.F_MM = system.Shell(
            core=self.__FL_iteration(self.V_ML, self.G, self.H_LM),
            input_layer=nn.Sequential(
                dsp.Transform(lambda x: x.diag_embed()), dsp.FFT(self.nfft)
            ),
        )
        self.set_G_to_GBI()

    # ================================== FORWARD PATH ==================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes one iteration of the feedback loop.

            **Args**:
                x (torch.Tensor): input signal.

            **Returns**:
                torch.Tensor: Depending on the input, it can be the microphones signals or the feedback loop matrix.

            **Usage**:
                If x is a vector of unit impulses of size (_, n_M), the output is a vector of size (_, n_M) representing the microphones signals.
                If x is a diagonal matrix of unit impulses of size (_, n_M, n_M), the output is a matrix of size (_, n_M, n_M) representing the feedback loop matrix.
                The first dimension of vectors and matrices depends on input_layer and output_layer of the Shell instance self.F_MM.
        """
        return self.F_MM(x)

    # ================================== OTHER METHODS ==================================

    # ------------------------ General gain methods ------------------------

    def get_G(self) -> nn.Module:
        r"""
        Return the general gain value in linear scale.

            **Returns**:
                torch.Tensor: general gain value (linear scale).
        """
        return self.G

    def set_G(self, g: float) -> None:
        r"""
        Set the general gain value in linear scale.

            **Args**:
                g (float): new general gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a float."
        self.G.assign_value(g * torch.ones(self.n_L))

    def get_current_GBI(self) -> torch.Tensor:
        r"""
        Return the Gain Before Instability (GBI) value in linear scale.
        The GBI is always computed with respect to a system general gain G=1.

            **Returns**:
                torch.Tensor: GBI value (linear scale).
        """
        # Save current G value
        g_current = self.G.param.data[0].clone()

        # Reset G module
        self.G.assign_value(torch.ones(self.n_L))

        # Compute the gain before instability
        maximum_eigenvalue = torch.max(get_magnitude(self.get_F_MM_eigenvalues()))
        gbi = torch.reciprocal(maximum_eigenvalue)

        # Restore G value
        self.set_G(g_current)

        return gbi

    def set_G_to_GBI(self) -> None:
        r"""
        Set the system general gain to match the current system GBI in linear scale.
        """
        # Compute the gain before instability
        gbi = self.get_current_GBI()

        # Apply gbi to the module
        self.set_G(gbi)

    # ------------------------------------------------------------------------------
    # ---------------------------- Virtual Room methods ----------------------------

    def normalize_U(self, value: float = 1.0) -> None:
        r"""
        Normalize the dsp matrix IRs to a Frobenius norm of given value.

            **Args**:
                value (float, optional): value to normalize the matrix IRs. Defaults to 1.0.
        """
        self.V_ML["U"].assign_value(
            self.V_ML["U"].param.data
            / torch.norm(self.V_ML["U"].param.data, "fro")
            * value
        )

    # ------------------------------------------------------------------------------
    # ------------------------ Feedback-loop matrix methods ------------------------

    def __FL_iteration(
        self, v_ml: OrderedDict, g: nn.Module, h_lm: nn.Module
    ) -> nn.Sequential:
        r"""
        Generate a Series object instance representing one iteration of the feedback loop.

            **Args**:
                - h_lm (nn.Module): Feedback paths from loudspeakers to microphones.
                - v_ml (OrderedDict): Virtual room components.
                - g (nn.Module): General gain.

            **Returns**:
                nn.Sequential: Series implementing one feedback-loop iteration.
        """
        f_mm = nn.Sequential()
        for key, value in v_ml.items():
            f_mm.add_module(key, value)

        f_mm.add_module("G", g)
        f_mm.add_module("H_LM", h_lm)

        return system.Series(f_mm)

    def get_F_MM_eigenvalues(self) -> torch.Tensor:
        r"""
        Compute the eigenvalues of the feedback-loop matrix.

            **Returns**:
                torch.Tensor: eigenvalues.
        """
        with torch.no_grad():

            # Compute eigenvalues
            evs = get_eigenvalues(
                self.F_MM.get_freq_response(fs=self.fs, identity=True)
            )

        return evs

    # ------------------------------------------------------------------------------
    # ---------------------------- Full system methods -----------------------------

    def __create_system(self) -> tuple[system.Shell, system.Shell]:
        f"""
        Create the full system's Natural and Electroacoustic paths.

            **Returns**:
                tuple[Shell, Shell]: Natural and Electroacoustic paths as Shell object instances.
        """
        # Build digital signal processor
        processor = nn.Sequential()
        for key, value in self.V_ML.items():
            processor.add_module(key, value)
        processor.add_module("G", self.G)
        # Build feedback loop
        feedback_loop = system.Recursion(fF=processor, fB=self.H_LM)
        # Build the electroacoustic path
        ea_components = nn.Sequential(
            OrderedDict(
                [
                    ("H_SM", self.H_SM),
                    ("FeedbackLoop", feedback_loop),
                    ("H_LA", self.H_LA),
                ]
            )
        )
        ea_path = system.Shell(
            core=ea_components,
            input_layer=dsp.FFT(self.nfft),
            output_layer=dsp.iFFT(self.nfft),
        )
        # Build the natural path
        nat_path = system.Shell(
            core=self.H_SA,
            input_layer=dsp.FFT(self.nfft),
            output_layer=dsp.iFFT(self.nfft),
        )
        return nat_path, ea_path

    def system_simulation(self) -> torch.Tensor:
        r"""
        Simulate the full system. Produces the system impulse response.

            **Returns**:
                torch.Tensor: system impulse response.
        """
        with torch.no_grad():

            # Generate the paths
            nat_path, ea_path = self.__create_system()
            # Compute system response
            y = nat_path.get_time_response() + ea_path.get_time_response()

        return y


# ============================================== Plots =============================================


def plot_evs_distributions(
    evs_1: torch.Tensor,
    evs_2: torch.Tensor,
    fs: int,
    nfft: int,
    label1: str = "Initialized",
    label2: str = "Optimized",
) -> None:
    r"""
    Plot the magnitude distribution of the given eigenvalues.

        **Args**:
            evs_init (torch.Tensor): First set of eigenvalues to plot.
            evs_opt (torch.Tensor): Second set of eigenvalues to plot.
            fs (int): Sampling frequency.
            nfft (int): FFT size.
            label1 (str, optional): Label for the first set of eigenvalues. Defaults to 'Initialized'.
            label2 (str, optional): Label for the second set of eigenvalues. Defaults to 'Optimized'.
    """
    idx1 = int(nfft / fs * 20)
    idx2 = int(nfft / fs * 20000)
    evs = mag2db(
        torch.cat((evs_1.unsqueeze(-1), evs_2.unsqueeze(-1)), dim=len(evs_1.shape))[
            idx1:idx2, :, :
        ]
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 20,
            "font.weight": "heavy",
            "text.usetex": True,
        }
    )
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(1, 1, 1)
    colors = ["tab:blue", "tab:orange"]
    for i in range(evs.shape[2]):
        evst = torch.reshape(evs[:, :, i], (evs.shape[0] * evs.shape[1], -1)).squeeze()
        evst_max = torch.max(evst, 0)[0]
        ax.boxplot(
            evst.numpy(),
            positions=[i],
            widths=0.7,
            showfliers=False,
            notch=True,
            patch_artist=True,
            boxprops=dict(edgecolor="k", facecolor=colors[i]),
            medianprops=dict(color="k"),
        )
        ax.scatter(
            [i], [evst_max], marker="o", s=35, edgecolors="black", facecolors=colors[i]
        )
    plt.ylabel("Magnitude in dB")
    plt.xticks([0, 1], [label1, label2])
    plt.xticks(rotation=90)
    ax.yaxis.grid(True)
    plt.title("Eigenvalue Magnitude Distribution")
    plt.tight_layout()


def plot_spectrograms(
    y_1: torch.Tensor,
    y_2: torch.Tensor,
    fs: int,
    nfft: int = 2**10,
    label1="Initialized",
    label2="Optimized",
    title="System Impulse Response Spectrograms",
) -> None:
    r"""
    Plot the spectrograms of the system impulse responses at initialization and after optimization.

        **Args**:
            - y_1 (torch.Tensor): First signal to plot.
            - y_2 (torch.Tensor): Second signal to plot.
            - fs (int): Sampling frequency.
            - nfft (int, optional): FFT size. Defaults to 2**10.
            - label1 (str, optional): Label for the first signal. Defaults to 'Initialized'.
            - label2 (str, optional): Label for the second signal. Defaults to 'Optimized'.
            - title (str, optional): Title of the plot. Defaults to 'System Impulse Response Spectrograms'.
    """
    Spec_init, f, t = mlab.specgram(
        y_1.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=nfft // 8
    )
    Spec_opt, _, _ = mlab.specgram(
        y_2.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=nfft // 8
    )

    max_val = max(Spec_init.max(), Spec_opt.max())
    Spec_init = Spec_init / max_val
    Spec_opt = Spec_opt / max_val

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 20,
            "font.weight": "heavy",
            "text.usetex": True,
        }
    )
    fig, axes = plt.subplots(
        2, 1, sharex=False, sharey=True, figsize=(7, 5), constrained_layout=True
    )

    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, 10 * np.log10(Spec_init), cmap="magma", vmin=-100, vmax=0)
    plt.ylim(20, 20000)
    plt.yscale("log")
    plt.title(label1)

    plt.subplot(2, 1, 2)
    im = plt.pcolormesh(t, f, 10 * np.log10(Spec_opt), cmap="magma", vmin=-100, vmax=0)
    plt.ylim(20, 20000)
    plt.yscale("log")
    plt.title(label2)

    fig.supxlabel("Time in seconds")
    fig.supylabel("Frequency in Hz")
    fig.suptitle(title)

    cbar = fig.colorbar(im, ax=axes[:], aspect=20)
    cbar.set_label("Magnitude in dB")
    ticks = np.arange(-100, 1, 20)
    cbar.ax.set_ylim(-100, 0)
    cbar.ax.set_yticks(ticks, ["-100", "-80", "-60", "-40", "-20", "0"])


# ==================================================================================================
# =========================================== Auxiliary ============================================


class AA_RIRs(object):
    def __init__(
        self, dir: str, n_S: int, n_L: int, n_M: int, n_A: int, fs: int
    ) -> None:
        r"""
        Room impulse response wrapper class.
        These room impulse responses were measured in the listening room called Otala inside
        the Aalto Acoustics Lab in the Aalto University's Otaniemi campus, Espoo, Finland.

            **Args**:
                - dir (str): Path to the room impulse responses.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        object.__init__(self)
        assert n_S == 1, "Only one source is supported."
        assert n_L <= 13, "Only up to 13 loudspeakers are supported."
        assert n_M <= 4, "Only up to 4 microphones are supported."
        assert n_A == 1, "Only one audience member is supported."

        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A
        self.fs = fs
        self.dir = dir
        self.__RIRs = self.__load_rirs()
        self.RIR_length = self.__RIRs.shape[0]

    def __load_rirs(self) -> torch.Tensor:
        r"""
        Give the directory, loads the corresponding RIRs.

            **Returns**:
                torch.Tensor: RIRs. dtype=torch.float32, shape = (15000, n_M, n_L).
        """

        rirs = torch.zeros(15000, 5, 13)

        lds_index = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        m = list(range(1, 5 + 1))
        l = lds_index[0:13]

        for mcs in range(5):
            for lds in range(13):
                w, sr = torchaudio.load(f"{self.dir}/mic{m[mcs]}_speaker{l[lds]}.wav")
                rirs[:, mcs, lds] = w[0, 0:15000].squeeze()

        if self.fs != sr:
            rirs = torchaudio.transforms.Resample(sr, self.fs)(rirs)

        rirs[:, 1, :] = rirs[:, 1, :] * db2mag(
            6
        )  # TODO: retake measurements, solve mic gain problems
        rirs[:, 3, :] = rirs[:, 3, :] * db2mag(-2)

        return rirs / (torch.norm(rirs, "fro"))  # TODO: choose if and how to normalize

    def get_scs_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the sources to microphones RIRs

            **Returns**:
                torch.Tensor: Sources to microphones RIRs. shape = (15000, n_M, n_S).
        """
        return self.__RIRs[:, 0 : self.n_M, 2].unsqueeze(2)

    def get_scs_to_aud(self) -> torch.Tensor:
        r"""
        Returns the sources to audience RIRs

            **Returns**:
                torch.Tensor: Sources to audience RIRs. shape = (15000, n_A, n_S).
        """
        return self.__RIRs[:, -1, 2].unsqueeze(1).unsqueeze(2)

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to microphones RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to microphones RIRs. shape = (15000, n_M, n_L).
        """
        return self.__RIRs[:, 0 : self.n_M, 0 : self.n_L]

    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to audience RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to audience RIRs. shape = (15000, n_A, n_L).
        """
        return self.__RIRs[:, -1, 0 : self.n_L].unsqueeze(1)


class MSE_evs(nn.Module):
    def __init__(self, iter_num: int, freq_points: int):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__()

        self.iter_num = iter_num
        self.idxs = torch.randperm(freq_points)
        self.evs_per_iteration = torch.ceil(
            torch.tensor(freq_points / self.iter_num, dtype=torch.float)
        )
        self.max_index = freq_points
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.

            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:, idxs, :, :]))
        evs_true = y_true[:, idxs, :]
        mse = torch.mean(torch.square(evs_pred - evs_true))
        return mse

    def __get_indexes(self):
        r"""
        Get the indexes of the frequency-point subset.

            **Returns**:
                torch.Tensor: Indexes of the frequency-point subset.
        """
        # Compute indeces
        idx1 = np.min(
            [int(self.interval_count * self.evs_per_iteration), self.max_index - 1]
        )
        idx2 = np.min(
            [int((self.interval_count + 1) * self.evs_per_iteration), self.max_index]
        )
        idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
        # Update interval counter
        self.interval_count = (self.interval_count + 1) % (self.iter_num)
        return idxs


def save_model_params(model: system.Shell, filename: str = "parameters"):
    r"""
    Retrieves the parameters of a feedback delay network (FDN) from a given network and saves them in .mat format.

        **Parameters**:
            model (Shell): The Shell class containing the FDN.
            filename (str): The name of the file to save the parameters without file extension. Defaults to 'parameters'.
        **Returns**:
            dict: A dictionary containing the FDN parameters.
                - 'FIR_matrix' (ndarray): The FIR matrix.
                - 'WGN_reverb' (ndarray): The WGN reverb.
                - 'G' (ndarray): The general gain.
                - 'H_LM' (ndarray): The loudspeakers to microphones RIRs.
                - 'H_LA' (ndarray): The loudspeakers to audience RIRs.
                - 'H_SM' (ndarray): The sources to microphones RIRs.
                - 'H_SA' (ndarray): The sources to audience RIRs.
    """

    param = {}
    param["FIR_matrix"] = model.V_ML["U"].param.squeeze().detach().clone().numpy()
    param["WGN_reverb"] = model.V_ML["R"].param.squeeze().detach().clone().numpy()
    param["G"] = model.G.param.squeeze().detach().clone().numpy()
    param["H_LM"] = model.H_LM.param.squeeze().detach().clone().numpy()
    param["H_LA"] = model.H_LA.param.squeeze().detach().clone().numpy()
    param["H_SM"] = model.H_SM.param.squeeze().detach().clone().numpy()
    param["H_SA"] = model.H_SA.param.squeeze().detach().clone().numpy()

    scipy.io.savemat(os.path.join(args.train_dir, filename + ".mat"), param)

    return param


# ==================================================================================================
# ============================================ Example =============================================


def example_AA(args) -> None:
    r"""
    Active Acoustics training test function.
    Training results are plotted showing the difference in performance between the initialized model and the optimized model.
    The model parameters are saved to file.
    You can modify the number of microphones (should be set between 1 and 4) and the number of loudspeakers (should be set between 1 and 13).
    Please use n_S = 1 and  n_A = 1.
    Measured room impulse responses for additional source and/or audience positions are not available.

        **Args**:
            A dictionary or object containing the necessary arguments for the function.
    """

    # --------------------- Parameters ------------------------
    samplerate = 48000  # Sampling frequency
    nfft = 96000  # FFT size
    microphones = 4  # Number of microphones
    loudspeakers = 13  # Number of loudspeakers
    FIR_order = 100  # FIR filter order
    wgn_RT = 1.0  # Reverberation time of the WGN reverb

    # ------------------- Model Definition --------------------
    model = AA(
        n_S=1,
        n_M=microphones,
        n_L=loudspeakers,
        n_A=1,
        fs=samplerate,
        nfft=nfft,
        FIR_order=FIR_order,
        wgn_RT=wgn_RT,
        alias_decay_db=-20,
    )

    # ------------- Performance at initialization -------------
    # Normalize for fair comparison
    model.normalize_U()
    # We initialize the model to an instable state.
    gbi_init = model.get_current_GBI()
    model.set_G(db2mag(mag2db(gbi_init) + 0))
    # Performance metrics
    evs_init = model.get_F_MM_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)

    # Save the model parameters
    save_model_params(model, filename="AA_parameters_init")

    # ----------------- Initialize dataset --------------------
    dataset = DatasetColorless(
        input_shape=(args.batch_size, nfft // 2 + 1, microphones),
        target_shape=(args.batch_size, nfft // 2 + 1, microphones),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(
        dataset, batch_size=args.batch_size, split=args.split, shuffle=False
    )

    # ------------- Initialize training process ---------------
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device,
    )
    criterion = MSE_evs(iter_num=args.num, freq_points=nfft // 2 + 1)
    trainer.register_criterion(criterion, 1)

    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------ Performance after optimization ------------
    # Normalize for fair comparison
    model.normalize_U()
    # Performance metrics
    evs_opt = model.get_F_MM_eigenvalues().squeeze(0)
    ir_opt = model.system_simulation().squeeze(0)

    # Save the model parameters
    save_model_params(model, filename="AA_parameters_optim")

    # ------------------------ Plots -------------------------
    plot_evs_distributions(
        get_magnitude(evs_init), get_magnitude(evs_opt), samplerate, nfft
    )
    plot_spectrograms(ir_init, ir_opt, samplerate)
    plt.show()

    return None


###########################################################################################

if __name__ == "__main__":

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()

    # ----------------------- Dataset ----------------------
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--num", type=int, default=2**8, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to use for computation"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="split ratio for training and validation",
    )
    # ---------------------- Training ----------------------
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="maximum number of epochs"
    )
    parser.add_argument(
        "--patience_delta",
        type=float,
        default=0.01,
        help="Minimum improvement in validation loss to be considered as an improvement",
    )
    # ---------------------- Optimizer ---------------------
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # ----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments
    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )

    # Run examples
    example_AA(args)
