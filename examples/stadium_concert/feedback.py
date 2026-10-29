"""Feedback of the singer's microphone: how much gain it allows before the loudspeakers
that play it make it ring, at which frequencies it rings first, and how that sounds.

    python feedback.py [scene.yaml] [--rays N] [--out feedback.wav]

The singer holds a supercardioid microphone 5 cm from the mouth, aimed at it, so its
rear faces the wedges. Every loudspeaker that plays the voice feeds back into it: all
but the subwoofers, the drum fill and those left out of simulations, at the scene's
levels, EQ and delays, so the monitors and the PA keep the balance the scene gives
them. The script simulates their responses at the microphone in first-order Ambisonics
and turns them into the microphone's pattern; their sum, after 2 ms of latency in the
console, is the loop.

It prints the potential acoustic gain (PAG) in each octave band: how much louder the
system can make the voice than the voice alone is, at the mix position and at the
singer's ears, before the loop reaches unity gain and rings on its own. The voice alone
is a talker's (subcardioid) from the singer's mouth, 15 cm from the ears. First as if
the system were rung out, each band's peaks taken down to its average; then as it
stands, where the loop's highest peak rings first and sets the gain for every band.
Last it renders what the microphone picks up when a burst of noise is sung into it,
with the loop 6 dB below, 2 dB below and 1 dB above unity gain: the first dies away,
the second rings on and the third howls, up to where the amplifiers clip.

The peaks come from the early paths' exact delays and phases and from a reverberant
tail whose phases are random: they are where a moment of this stage would ring, not
where a given night's would.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, simulate
from acoustic_spaces.audio import write_wav
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import octave_filter_gains
from acoustic_spaces.scene import Listener, Source
from acoustic_spaces.spatial.ambisonics import Ambisonics, real_spherical_harmonics

#: The loudspeakers the voice is not sent to.
NOT_THE_VOICE = ("subs", "drum_fill")
#: The singer's mouth from the middle of the head (m, forward and up); the microphone's
#: distance from it and its pattern (a + (1 - a) cos theta: a supercardioid); and the
#: distance from the mouth to the ears.
MOUTH = (0.09, -0.08)
MIC_DISTANCE, MIC_PATTERN = 0.05, 0.366
EARS = 0.15
#: The voice's pattern, a talker's, as a + (1 - a) cos theta (a subcardioid).
VOICE, VOICE_PATTERN = "subcardioid", 0.7
#: The console's latency (s); the loop gains rendered (dB re unity gain), each for
#: RENDER_S, the burst sung into the microphone (s), and where the amplifiers clip
#: (re the burst's level, dB).
LATENCY_S = 0.002
RENDERED_DB = (-6.0, -2.0, 1.0)
RENDER_S, BURST_S, CLIP_DB = 3.0, 0.3, 12.0
RATE = 48_000


def _voice_energy(distance: float, cos_angle: float) -> float:
    """The voice's energy per unit sound power ``distance`` from the mouth in free
    field, ``cos_angle`` off its aim."""
    a = VOICE_PATTERN
    q = (a + (1.0 - a) * cos_angle) ** 2 / (a**2 + (1.0 - a) ** 2 / 3.0)
    return q / (4.0 * math.pi * distance**2)


def _bands(n_fft: int) -> list[np.ndarray]:
    """The FFT bins of each octave band."""
    f = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    return [(f >= fm / math.sqrt(2)) & (f < fm * math.sqrt(2)) for fm in OCTAVE_BANDS_HZ]


def loop(
    room: Room, singer: str = "singer", rays: int | None = None, seed: int | None = None
) -> dict:
    """The loop of ``singer``'s microphone: ``response``, the loudspeakers' response at
    the microphone (samples at RATE, latency included); ``pag``, the potential acoustic
    gain per octave band at the mix position and at the singer's ears (dB), rung out;
    ``peaks``, how far the loop's highest peak stands above each band's average (dB);
    ``ring_hz``, the frequencies of the three highest peaks, highest first; and
    ``speakers``, the loudspeakers in the loop. ``rays`` sets the late stage's rays, and
    ``seed`` repeats a run's random rays."""
    scene = room.scene
    head = next(listener for listener in scene.listeners if listener.name == singer)
    turn = np.array(head.orientation.rotation())
    aim = turn[:, 0]
    mouth = np.asarray(head.position) + MOUTH[0] * aim + MOUTH[1] * turn[:, 2]
    mic = Listener(name="microphone", position=tuple(mouth + MIC_DISTANCE * aim))
    mix = next(listener for listener in scene.listeners if listener.name == "mix_position")
    voice = Source(
        name="voice", position=tuple(mouth), orientation=head.orientation, directivity=VOICE
    )
    speakers = [s for s in scene.sources if s.in_use and s.name not in NOT_THE_VOICE]
    settings = scene.simulation
    if rays:
        settings = settings.model_copy(
            update={"late": settings.late.model_copy(update={"rays": rays})}
        )
    if seed is not None:
        settings = settings.model_copy(update={"seed": seed})
    # The voice is heard alone at the mix position only: the microphone and the ears are
    # too near the mouth for paths, and its level at the ears is the free field's.
    copy = scene.model_copy(update={"sources": speakers, "listeners": [mic, mix, head]})
    result = simulate(Room.from_scene(copy, reuse=room), settings=settings)
    copy = scene.model_copy(update={"sources": [voice], "listeners": [mix]})
    heard = simulate(Room.from_scene(copy, reuse=room), settings=settings)
    # The microphone's pattern from first-order Ambisonics (ACN / SN3D), aimed at the mouth.
    harmonics = real_spherical_harmonics(1, -aim[None])[0]
    pattern = np.concatenate([[MIC_PATTERN], (1.0 - MIC_PATTERN) * harmonics[1:]])
    lag = int(round(LATENCY_S * RATE))
    parts, system = [], {"mix": 0.0, "ears": 0.0}
    for speaker in speakers:
        # The response holds the feed (EQ, delay); the sound power without it scales it.
        power = speaker.power_db() - speaker.feed_db()
        ir = result[speaker.name, mic.name].spatial_impulse_response(Ambisonics(1), RATE)
        parts.append(10.0 ** (power.mean() / 20.0) * (ir @ pattern))
        for where, listener in (("mix", mix), ("ears", head)):
            levels = result[speaker.name, listener.name].levels(speaker.power_db())
            system[where] = system[where] + 10.0 ** (levels / 10.0)
    response = np.zeros(lag + max(len(p) for p in parts))
    for part in parts:
        response[lag : lag + len(part)] += part
    # The loop's energy in each band, per unit sound power: an impulse response's direct
    # sound r m away is 1 / r, 4 pi times the free field's energy per unit power.
    n_fft = 1 << int(math.ceil(math.log2(len(response))))
    spectrum = np.abs(np.fft.rfft(response, n_fft)) ** 2
    weights = np.full(spectrum.shape, 2.0)
    weights[0] = weights[-1] = 1.0
    at_mic = octave_filter_gains(n_fft, RATE) ** 2 @ (weights * spectrum) / n_fft
    at_mic = at_mic / (4.0 * math.pi)
    alone = {
        "mix": 10.0 ** (heard["voice", mix.name].levels(np.zeros(len(OCTAVE_BANDS_HZ))) / 10),
        "ears": np.full(len(OCTAVE_BANDS_HZ), _voice_energy(EARS, 0.0)),
    }
    at_mouth = _voice_energy(MIC_DISTANCE, 1.0)
    # At unity gain the loudspeakers give the listener, for the voice at the microphone,
    # the system's energy there over its energy at the microphone.
    pag = {
        where: 10.0 * np.log10(system[where] * at_mouth / (at_mic * alone[where]))
        for where in system
    }
    # Within each band the loop is uneven: its highest peak rings first.
    in_band = _bands(n_fft)
    highest = max(spectrum[b].max() for b in in_band)
    peaks = np.array([10.0 * np.log10(highest / spectrum[b].mean()) for b in in_band])
    f = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    audible = (f >= OCTAVE_BANDS_HZ[0] / math.sqrt(2)) & (f < OCTAVE_BANDS_HZ[-1] * math.sqrt(2))
    ring = []
    for k in np.argsort(spectrum * audible)[::-1]:
        # Peaks a third of an octave apart or more.
        if audible[k] and all(abs(math.log2(f[k] / r)) >= 1 / 3 for r in ring):
            ring.append(float(f[k]))
        if len(ring) == 3:
            break
    return {
        "response": response,
        "pag": pag,
        "peaks": peaks,
        "ring_hz": ring,
        "speakers": [s.name for s in speakers],
    }


def render(response: np.ndarray, gain_db: float, seconds: float = RENDER_S) -> np.ndarray:
    """What the microphone picks up (samples at RATE) when a burst of pink noise is sung
    into it, with the loop ``gain_db`` from unity gain at its highest peak; the
    amplifiers clip where the microphone picks up CLIP_DB more than the burst."""
    n_fft = 1 << int(math.ceil(math.log2(len(response) + RATE // 10)))
    gain = 10.0 ** (gain_db / 20.0) / np.abs(np.fft.rfft(response, n_fft)).max()
    # A partitioned convolution in blocks no longer than the loop's shortest way round,
    # so each block of the microphone's signal depends only on the ones before it.
    first = int(np.argmax(np.abs(response) > 1e-6 * np.abs(response).max()))
    block = 1 << int(math.floor(math.log2(first)))
    parts = -(-len(response) // block)
    padded = np.zeros(parts * block)
    padded[: len(response)] = response
    filters = np.fft.rfft(padded.reshape(parts, block), 2 * block, axis=1)[1:]
    total = int(seconds * RATE)
    rng = np.random.default_rng(1)
    white = np.fft.rfft(rng.standard_normal(int(BURST_S * RATE)))
    f = np.fft.rfftfreq(int(BURST_S * RATE), 1.0 / RATE)
    burst = np.fft.irfft(white / np.sqrt(np.maximum(f, 20.0)), int(BURST_S * RATE))
    burst /= np.sqrt(np.mean(burst**2))
    voice = np.zeros(total + block)
    voice[: len(burst)] = burst
    clip = 10.0 ** (CLIP_DB / 20.0)
    # The spectra of the blocks fed to the loudspeakers, each with the one before it,
    # latest first.
    fed = np.zeros((len(filters), block + 1), dtype=complex)
    before = np.zeros(block)
    out = np.zeros(total + block)
    for start in range(0, total, block):
        back = np.fft.irfft((fed * filters).sum(axis=0), 2 * block)[block:]
        out[start : start + block] = voice[start : start + block] + back
        now = gain * clip * np.tanh(out[start : start + block] / clip)
        fed = np.roll(fed, 1, axis=0)
        fed[0] = np.fft.rfft(np.concatenate([before, now]))
        before = now
    return out[:total]


#: Its name in the editor's Run demo window.
TITLE = "Feedback of the singer's microphone"


def parser() -> argparse.ArgumentParser:
    """Its arguments, which the editor's Run demo window asks for too."""
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument(
        "scene", nargs="?", default=Path(__file__).with_name("scene.yaml"), help="the scene"
    )
    found.add_argument("--rays", type=int, help="rays per band (fewer is quicker)")
    found.add_argument(
        "--out", type=Path, default=Path("feedback.wav"), help="the rendered loop (WAV)"
    )
    return found


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    found = loop(Room.from_yaml(args.scene), rays=args.rays)
    bands = " ".join(f"{f if f < 1000 else f'{f // 1000}k':>5}" for f in OCTAVE_BANDS_HZ)
    print(f"The singer's microphone feeds {len(found['speakers'])} loudspeakers.")
    for title, less in (("rung out", 0.0), ("as it stands", found["peaks"])):
        print(f"\nPAG, {title} (dB) {bands}")
        for where, label in (("mix", "at the mix position"), ("ears", "at the singer's ears")):
            row = found["pag"][where] - less
            print(f"  {label:20} " + " ".join(f"{v:5.1f}" for v in row))
    ring = ", ".join(f"{f:.0f} Hz" for f in found["ring_hz"])
    print(f"\nIt rings first at {ring}.")
    parts = [render(found["response"], gain) for gain in RENDERED_DB]
    signal = np.concatenate(parts)
    write_wav(args.out, 0.5 * signal / np.abs(signal).max(), RATE)
    steps = ", ".join(f"{g:+.0f} dB" for g in RENDERED_DB)
    print(f"Wrote {args.out}: the loop at {steps} from unity gain, {RENDER_S:g} s each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
