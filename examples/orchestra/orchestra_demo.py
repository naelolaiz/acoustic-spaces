"""Beethoven's 8th symphony in the concert hall example, heard from its four seats, and a
video that tours them through three movements.

The recording is the TU Berlin's anechoic one (Böhm, Ackermann and Weinzierl, Orchester
Wiener Akademie; CC BY 4.0, https://doi.org/10.14279/depositonce-6729.2): 61 tracks, one
per player, which the `orchestra` example puts on the stage as 61 sources. The three
movements it has (I, II and IV) come as movement1.zip, movement2.zip and movement4.zip:
`tracks` reads the 61 tracks out of each on the TU Berlin's server, by HTTP range
requests, without keeping the zip (2.0 GB for the three), or out of a copy of the zip
downloaded before (`--zips FOLDER`, the example's folder by default). Run this from the
folder of the example (`acoustic-spaces examples --copy . orchestra && cd orchestra`),
in the Python environment acoustic-spaces is installed in, with ffmpeg on the PATH:

    python orchestra_demo.py tracks    # download the 61 tracks of each movement (2.0 GB)
    python orchestra_demo.py simulate  # acoustic-spaces simulate scene.yaml --out results/binaural
    python orchestra_demo.py render    # each stop of the tour: a movement at a seat, binaural
    python orchestra_demo.py video     # demo/orchestra_binaural.mp4, its audio, the stills

or `python orchestra_demo.py all`, which runs them in turn and leaves out what is there
already (the tracks, the simulation). The simulation goes to ./results (--results), in a
folder for each output (results/binaural); the tracks go to ./demo (--work), and each
run's mixes and stills to a folder there named after its settings
(demo/orchestra_binaural/), with the video, its audio and its thumbnail beside it, so
that a run with other settings writes over none of them. Use headphones;
`--hrtf me.sofa` plays through your own or measured HRTFs instead of a spherical head
(the simulation takes it too). `--input fl1=flute.wav` has a player play a file of yours
instead of their track, in every movement of the tour, at the level of that track. The
editor runs the same steps from Simulation > Run demo…

The tour plays what the recording has of the three movements (70 to 85 s of each), each
from two to four of the seats in turn:
the first from the middle of the stalls, the front rows and the conductor's podium, the
second from the back of the hall and the middle, the finale from the podium, the front,
the back and the middle again. Within a movement the seat changes at the quietest moment
near each third (or half, or quarter) of it, so the cuts fall between phrases, with a 1 s
crossfade; between movements the hall's reverberation dies away and there is a pause
(the video runs about four minutes).
Every seat's mix is shifted earlier by the time the orchestra's sound takes to reach it
(10 ms on the podium, 85 ms at the back), so the music runs on across a cut. All stops
keep one gain (the loudest peak at -1 dBFS): the hall's differences are heard as the
simulation gives them, from the podium, 2 m from the first violins, to the back row,
and the captions give each stop's level against the loudest. Every source has the same
power, so the balance between the instruments is the recording's own.
`--listeners stalls-rear,conductor` plays each movement at the seats of its tour among
those, in that order (the finale from the back, then the podium; a `--tour` given keeps
its own order, `--listeners` leaving out the others), or at all of them when its tour
has none of them. `--sources Winds` plays the winds alone (by section,
`Strings/Violins I` to `Percussion/Timpani`, or by player, `fl1`), in a run of its own
(demo/orchestra_Winds_binaural/); the stills, their credit and the thumbnail count and
light up those players only. A movement every picked player rests through (the
trumpets and the timpani in II) is left out of the tour, unless `--input` gives one of
them a recording: it plays that file in every movement. `--movements 4,1` plays the
finale, then the first movement.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import yaml

from acoustic_spaces import fetch
from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Source,
    Step,
    chosen_order,
    common_options,
    given_names,
    input_option,
    listener_names,
    listeners_in,
    listeners_option,
    picked,
    pipeline,
    playing,
    product,
    run_folder,
    sibling,
    sources_options,
)
from acoustic_spaces.demos.audio import envelope, mmss, quiet_moment, read, write

#: The movements the recording has: numeral and title, and the zip each comes in.
MOVEMENTS = {
    1: ("I", "Allegro vivace e con brio", "movement1"),
    2: ("II", "Allegretto scherzando", "movement2"),
    4: ("IV", "Allegro vivace", "movement4"),
}
#: Each movement's zip on the TU Berlin's server (DepositOnce), which answers range
#: requests.
BITSTREAMS = "https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/"
ZIPS = {
    1: BITSTREAMS + "9fc848cb-33ec-435f-a6ea-bb73294a01b8/content",
    2: BITSTREAMS + "5e704ff9-53c0-4378-a8ef-5d2e6cdc8388/content",
    4: BITSTREAMS + "2b54758e-148e-439d-b687-2a960d704611/content",
}
#: Bytes a range request reads of a zip: the tracks lie one after the other in it, and
#: the server takes about half a second to answer each request (in blocks of 4 MiB the
#: three movements took 622 s, in these 189 s).
BLOCK = 32 << 20
#: MB the 61 tracks of each movement take to download in those blocks, as measured:
#: nearly the whole zip, as the blocks of the last tracks hold the movement's mix.
DOWNLOAD_MB = {1: 642, 2: 605, 4: 743}
#: Source name = track file, as in auralize.sh.
TRACKS = {
    "fl1": "01_Fl_1.wav",
    "fl2": "02_Fl_2.wav",
    "ob1": "03_Ob_1.wav",
    "ob2": "04_Ob_2.wav",
    "cl1": "05_Cl_1.wav",
    "cl2": "06_Cl_2.wav",
    "bn1": "07_Bn_1.wav",
    "bn2": "08_Bn_2.wav",
    "hn1": "09_Hrn_1.wav",
    "hn2": "10_Hrn_2.wav",
    "tp1": "11_Trp_1.wav",
    "tp2": "12_Trp_2.wav",
    "timpani": "13_Tp.wav",
    **{
        f"vl1_{k:02d}": f"{13 + k}_Vl_1_{1 + (k - 1) // 7}_{1 + (k - 1) % 7}.wav"
        for k in range(1, 15)
    },
    **{
        f"vl2_{k:02d}": f"{27 + k}_Vl_2_{1 + (k - 1) // 6}_{1 + (k - 1) % 6}.wav"
        for k in range(1, 7)
    },
    **{f"vl2_{k:02d}": f"{27 + k}_Vl_2_2_{k - 6}.wav" for k in range(7, 12)},
    **{f"vla_{k:02d}": f"{38 + k}_Vla_1_{k}.wav" for k in range(1, 7)},
    **{f"vla_{k:02d}": f"{38 + k}_Vla_2_{k - 6}.wav" for k in range(7, 12)},
    **{f"vc_{k}": f"{49 + k}_Vc_{k}.wav" for k in range(1, 7)},
    **{f"db_{k}": f"{55 + k}_Db_{k}.wav" for k in range(1, 7)},
}
#: The sections of the orchestra, in the order ``--sources`` lists them: the prefix of
#: their players' names, their group (``--sources Strings`` picks the five groups under
#: it, ``--sources Winds/Flutes`` the flutes) and what the stills call them in a
#: sentence. The scene has no groups (its simulation stays as it is): the demo's own.
SECTIONS = [
    ("vl1_", "Strings/Violins I", "first violins"),
    ("vl2_", "Strings/Violins II", "second violins"),
    ("vla_", "Strings/Violas", "violas"),
    ("vc_", "Strings/Cellos", "cellos"),
    ("db_", "Strings/Double basses", "double basses"),
    ("fl", "Winds/Flutes", "flutes"),
    ("ob", "Winds/Oboes", "oboes"),
    ("cl", "Winds/Clarinets", "clarinets"),
    ("bn", "Winds/Bassoons", "bassoons"),
    ("hn", "Brass/Horns", "horns"),
    ("tp", "Brass/Trumpets", "trumpets"),
    ("timpani", "Percussion/Timpani", "timpani"),
]
#: The players who rest through a whole movement: the Allegretto scherzando is scored
#: without trumpets and timpani. A run of only them leaves that movement out, unless
#: ``--input`` gives one of them a recording, which it plays in every movement.
RESTS = {2: ("tp1", "tp2", "timpani")}
#: The default tour: the movements in order, each with the seats that hear it in turn,
#: for equal shares of it (the cuts moved to the quietest moment near each share's end).
TOUR = [
    (1, ["stalls-middle", "stalls-front", "conductor"]),
    (2, ["stalls-rear", "stalls-middle"]),
    (4, ["conductor", "stalls-front", "stalls-rear", "stalls-middle"]),
]
TITLES = {
    "conductor": ("The conductor's podium", "on the stage, 2 m from the first violins"),
    "stalls-front": ("Front of the stalls", "4 m from the stage's edge"),
    "stalls-middle": ("Middle of the stalls", "12 m from the stage, left of the aisle"),
    "stalls-rear": ("Back of the stalls", "the last rows, 21 m from the stage"),
}
SHORT = {
    "conductor": "podium",
    "stalls-front": "front",
    "stalls-middle": "middle",
    "stalls-rear": "back",
}
#: How far (s) a cut may move from its share's end to a quieter moment.
QUIET_S = 10.0
CROSSFADE_S = 1.0
#: The pause between movements (s), after the hall has died away.
PAUSE_S = 3.0
FADE_OUT_S = 4.0

# --------------------------------------------------------------------------- the tracks


def section_of(name: str) -> tuple[str, str]:
    """The group of the player ``name`` (``Strings/Violins I``) and what the stills call
    their section (``first violins``)."""
    for prefix, group, called in SECTIONS:
        if name.startswith(prefix):
            return group, called
    raise ValueError(f"no section for the player {name!r}")


def sources(args) -> list[Source]:
    """The players ``--sources`` picks from, by name or by section (:data:`SECTIONS`),
    section by section, every one played by default."""
    groups = [group for _, group, _ in SECTIONS]
    order = sorted(TRACKS, key=lambda name: groups.index(section_of(name)[0]))
    return [Source(name, section_of(name)[0]) for name in order]


def who_plays(heard) -> tuple[str, bool]:
    """The players ``heard`` in words, and whether they are one: ``the orchestra`` for
    all of them, else each whole family (``the strings``) and section (``the flutes``)
    or how many of a section (``1 of the 2 flutes``), or for more than three of those,
    how many of the orchestra."""
    heard = set(heard)
    if heard >= set(TRACKS):
        return "the orchestra", True
    groups = {name: section_of(name)[0] for name in TRACKS}
    parts = []
    for family in dict.fromkeys(group.split("/")[0] for _, group, _ in SECTIONS):
        sections = [(g, called) for _, g, called in SECTIONS if g.startswith(family + "/")]
        everyone = {name for name, g in groups.items() if g.startswith(family + "/")}
        if len(sections) > 1 and everyone <= heard:
            parts.append(f"the {family.lower()}")
            continue
        for group, called in sections:
            members = [name for name, g in groups.items() if g == group]
            n = len(heard.intersection(members))
            if n == len(members):
                parts.append(f"the {called}")
            elif n:
                parts.append(f"{n} of the {len(members)} {called}")
    if len(parts) > 3:
        return f"{len(heard)} of the {len(TRACKS)} players", False
    words = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return words, len(parts) == 1 and parts[0].startswith("1 of ")


def movements_given(args) -> list[int] | None:
    """The movements ``--movements`` names, in its order (by number: 1, 2, 4), or None
    when it names none."""
    names = given_names(args.movements)
    if not names:
        return None
    return [int(name) for name in names]


def movements_of(args) -> list[int]:
    """The movements of the run's tour, in its order (:func:`stops_of`)."""
    return list(dict.fromkeys(movement for movement, _ in stops_of(args)))


def tracks_folder(args, movement: int) -> Path:
    return args.work / "tracks" / MOVEMENTS[movement][2]


def missing_tracks(args, movement: int) -> list[str]:
    """The tracks of ``movement`` not in the work folder yet."""
    folder = tracks_folder(args, movement)
    return [track for track in TRACKS.values() if not (folder / track).exists()]


def tracks_done(args) -> bool:
    """Whether every track of every movement of the tour is in the work folder."""
    return not any(missing_tracks(args, movement) for movement in movements_of(args))


def copy_of(args, movement: int) -> Path | None:
    """The movement's zip downloaded before, in ``--zips``: the zip, or the folder it was
    unpacked into (movementN/anechoic, with every track); None when neither is there."""
    name = MOVEMENTS[movement][2]
    unpacked = args.zips / name / "anechoic"
    if all((unpacked / track).exists() for track in TRACKS.values()):
        return unpacked
    zip_path = args.zips / f"{name}.zip"
    return zip_path if zip_path.exists() else None


def tracks_download(args) -> str | None:
    """What ``tracks`` downloads: the tracks of the tour's movements that are neither in
    the work folder nor in ``--zips``, and how much at most; None for nothing."""
    needed = [m for m in movements_of(args) if missing_tracks(args, m) and copy_of(args, m) is None]
    if not needed:
        return None
    numerals = [MOVEMENTS[m][0] for m in needed]
    which = " and ".join([", ".join(numerals[:-1]), numerals[-1]] if len(needed) > 1 else numerals)
    mb = sum(DOWNLOAD_MB[m] for m in needed)
    size = f"{mb} MB" if mb < 1000 else f"{mb / 1000:.1f} GB"
    if any(len(missing_tracks(args, m)) < len(TRACKS) for m in needed):
        size = "at most " + size  # a download cut short goes on where it stopped
    plural = "s" if len(needed) > 1 else ""
    return f"the 61 tracks of movement{plural} {which} from the TU Berlin's server ({size})"


def tracks(args) -> None:
    """The 61 tracks of each movement of the tour (not the zip's mix) into
    demo/tracks/movementN, read out of movementN.zip on the TU Berlin's server by range
    requests of :data:`BLOCK`, or out of the copy of it in ``--zips`` (:func:`copy_of`).
    Tracks there already are kept, so a download cut short goes on where it stopped."""
    for movement in movements_of(args):
        folder = tracks_folder(args, movement)
        missing = missing_tracks(args, movement)
        if not missing:
            print(f"{folder}: all {len(TRACKS)} tracks there")
            continue
        copy = copy_of(args, movement)
        if copy is not None and copy.is_dir():
            print(f"copying the tracks from {copy}")
            folder.mkdir(parents=True, exist_ok=True)
            for track in missing:
                part = folder / f"{track}.part"
                shutil.copyfile(copy / track, part)
                part.replace(folder / track)
        else:
            source = str(copy) if copy is not None else ZIPS[movement]
            print(f"reading {source}" + ("" if copy is not None else " (range requests)"))
            archive = zipfile.ZipFile(copy) if copy is not None else fetch.open_zip(source, BLOCK)
            with archive:
                members = {Path(name).name: name for name in archive.namelist()}
                absent = [track for track in missing if track not in members]
                if absent:
                    raise SystemExit(f"error: {source} has no {absent[0]}")
                for k, track in enumerate(missing, 1):
                    fetch.copy_member(archive, members[track], folder / track)
                    print(f"  {track} ({k} of {len(missing)})")
        print(f"wrote {len(missing)} tracks to {folder}")


# --------------------------------------------------------------------------- the mixes


def simulations(args) -> list[pipeline.Simulation]:
    """The simulation the demo mixes from; the scene names the outputs it writes (mono
    and binaural), so the demo does not; into the folder of the run's output in
    ``--results`` (:func:`~acoustic_spaces.demos.pipeline.results_for`)."""
    return [pipeline.Simulation(args.scene, pipeline.results_for(args.results, args))]


def responses_of(args) -> list[dict]:
    return pipeline.responses_of(simulations(args)[0], args)


def tour_given(args) -> list[tuple[int, str]]:
    """(movement, seat) for each stop of the tour, ``--tour``'s or the default one."""
    if not args.tour:
        return [(movement, seat) for movement, seats in TOUR for seat in seats]
    stops = []
    for part in args.tour.split(","):
        movement, seat = part.split(":")
        stops.append((int(movement), seat.strip()))
    return stops


def of_movements(args) -> list[tuple[int, str]]:
    """The stops of the tour (:func:`tour_given`) of the movements ``--movements`` names,
    in its order; all of them when it names none."""
    stops = tour_given(args)
    chosen = movements_given(args)
    if chosen is None:
        return stops
    return [(movement, seat) for m in chosen for movement, seat in stops if movement == m]


def stops_of(args) -> list[tuple[int, str]]:
    """(movement, seat) for each stop of the run's tour: ``--tour``'s or the default one,
    of the movements ``--movements`` names, in its order, and of those not every player
    that plays (``--sources``) rests through (:data:`RESTS`): a player ``--input`` gives
    a recording plays it in every movement, so it rests through none; at the seats
    ``--listeners`` names: each movement at the seats of its tour among them, in that
    option's order (``--tour``'s order, when it is given), or at all of them when its tour
    has none of them."""
    heard = set(playing(args, list(TRACKS)))
    fed = set(getattr(args, "inputs", None) or {})
    stops = [
        (m, seat) for m, seat in of_movements(args) if not heard <= set(RESTS.get(m, ())) - fed
    ]
    names = listener_names(args.listeners)
    if names is None:
        return stops
    kept = []
    for movement in dict.fromkeys(m for m, _ in stops):
        here = [seat for m, seat in stops if m == movement]
        seats = picked(args, here) if args.tour else chosen_order(args, here)
        kept += [(movement, seat) for seat in seats or names]
    return kept


def stops_problem(args) -> str | None:
    """What keeps the run from a tour (``args`` resolved): a movement ``--movements`` or
    ``--tour`` names that the recording lacks, a seat ``--tour`` names that the scene
    lacks, or no stop left of the tour."""
    have = ", ".join(str(m) for m in MOVEMENTS)
    try:
        chosen = movements_given(args)
    except ValueError:
        return f"--movements takes the movements' numbers ({have}), not {args.movements!r}"
    for movement in chosen or []:
        if movement not in MOVEMENTS:
            return f"--movements {movement}: the recording has the movements {have}"
    try:
        tour = tour_given(args)
    except ValueError:
        return f"--tour takes movement:seat,... (as 1:conductor), not {args.tour!r}"
    seats = listeners_in(args.scene) if Path(args.scene).is_file() else None
    for movement, seat in tour:
        if movement not in MOVEMENTS:
            return f"--tour {movement}:{seat}: the recording has the movements {have}"
        if seats is not None and seat not in seats:
            return f"--tour {movement}:{seat}: no such seat here; it has {', '.join(seats)}"
    stops = of_movements(args)
    if not stops:
        return f"--movements {args.movements}: the tour (--tour) plays none of them"
    if getattr(args, "playing", None) and not stops_of(args):  # none: --sources says why
        numerals = [MOVEMENTS[m][0] for m in dict.fromkeys(m for m, _ in stops)]
        return (
            f"--sources {args.sources}: the players it picks rest through movement "
            f"{', '.join(numerals)}, all the tour plays"
        )
    return None


def problems(demo: Demo, args) -> str | None:
    """The movements and the tour the run plays (:func:`stops_problem`)."""
    return stops_problem(demo.resolved(args))


def key_of(movement: int, seat: str) -> str:
    """How a stop's mix is named in the run's seats folder."""
    return f"{MOVEMENTS[movement][2]}_{seat}"


def mixes_of(args) -> list[pipeline.Mix]:
    """A mix per stop: the movement's tracks heard at the seat. Every source plays as
    loud as the next, so the arrival time counts them all alike. render_mixes keeps the
    players that play in the run alone (``--sources``), their feeds and their weights
    (:func:`~acoustic_spaces.demos.pipeline.only_playing`)."""
    simulation = simulations(args)[0]
    weights = dict.fromkeys(TRACKS, 0.0)
    heard = [TRACKS[name] for name in playing(args, list(TRACKS))]
    mixes = []
    for movement, seat in stops_of(args):
        folder = tracks_folder(args, movement)
        if not all((folder / track).exists() for track in heard):
            raise SystemExit(f"error: no tracks in {folder}; run: {DEMO.command('tracks', args)}")
        feeds = {name: folder / track for name, track in TRACKS.items()}
        mixes.append(
            pipeline.Mix(
                key_of(movement, seat),
                simulation,
                seat,
                feeds,
                weights,
                {"movement": movement, "seat": seat},
            )
        )
    return mixes


def plays(args) -> dict[str, str]:
    """What each player that plays in the run (``--sources``) plays unless ``--input``
    gives them a file: their track."""
    return {name: f"{TRACKS[name]} of each movement" for name in playing(args, list(TRACKS))}


def render(args) -> None:
    """Each stop's mix: a movement at a seat (``--format``, binaural by default), then
    one gain for all of them (the loudest peak at -1 dBFS)."""
    pipeline.render_mixes(mixes_of(args), run_folder(args) / "seats", args, inputs=args.inputs)


# --------------------------------------------------------------------------- the tour


def tour_of(args, seats: dict) -> list[pipeline.TourStop]:
    """The stops with their times: in each movement the seats take equal shares, each
    cut moved to the quietest moment within QUIET_S of its share's end (by the first
    seat's envelope); the movements follow each other with a pause between them, so the
    hall dies away before the next one."""
    stops = stops_of(args)
    folder = run_folder(args) / "seats"
    tour: list[pipeline.TourStop] = []
    t = 0.0
    for movement in dict.fromkeys(m for m, _ in stops):
        here = [seat for m, seat in stops if m == movement]
        first = key_of(movement, here[0])
        length = seats["seats"][first]["length_s"]
        env = envelope(read(pipeline.mix_path(folder, seats, first))[0], seats["rate"])
        cuts = [0.0]
        for k in range(1, len(here)):
            cuts.append(quiet_moment(env, 0.1, length * k / len(here), QUIET_S))
        cuts.append(length)
        for k, (seat, start, stop) in enumerate(zip(here, cuts[:-1], cuts[1:], strict=True)):
            tour.append(
                pipeline.TourStop(
                    key_of(movement, seat),
                    t + start,
                    start,
                    stop,
                    joined_before=k > 0,
                    joined_after=k < len(here) - 1,
                )
            )
        t += length + PAUSE_S
    return tour


def facts_of(stop: dict, heard) -> list[str]:
    """The lines under the seat on a stop's still: its level against the loudest stop's,
    and when the sound of the players ``heard`` reaches it (:func:`who_plays`)."""
    relative = stop["relative_db"]
    who, one = who_plays(heard)
    reaches = "reaches you" if one else "reach you"
    return [
        "The loudest stop of the tour"
        if relative > -0.05
        else f"{relative:.1f} dB(A) against the loudest stop",
        f"{who[0].upper()}{who[1:]} {reaches} {stop['lag_s'] * 1e3:.0f} ms after "
        f"{'it plays' if one else 'they play'}",
    ]


def video(args) -> None:
    """The run's audio (``<run>.wav``), one still per stop of the tour, and its video
    (``<run>.mp4``): the stills with a scrolling spectrogram in the panel at their foot
    (ffmpeg, unless ``--no-mp4``)."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    pictures = sibling(__file__, "make_images")
    folder = run_folder(args) / "seats"
    seats = pipeline.seats_of(folder, DEMO.command("render", args), args)
    keys = [key_of(movement, seat) for movement, seat in stops_of(args)]
    pipeline.missing_mixes(seats, keys, DEMO.command("render", args))
    tour = tour_of(args, seats)
    audio = pipeline.tour_audio(
        tour,
        folder,
        seats,
        crossfade_s=CROSSFADE_S,
        fade_out_s=FADE_OUT_S,
        scene=args.scene,
    )
    sound = product(args, ".wav")
    write(sound, audio, seats["rate"])
    print(f"wrote {sound} ({len(audio) / seats['rate']:.1f} s)")
    scene = yaml.safe_load(args.scene.read_text(encoding="utf-8"))
    report = pipeline.read_json(simulations(args)[0].results / pipeline.PARAMETERS)
    heard = playing(args, list(TRACKS))
    credit = pipeline.credited(pictures.CREDIT, seats, heard)
    replaced = len({name for v in seats["seats"].values() for name in v.get("inputs", {})})
    played = f"{len(heard) - replaced} anechoic tracks" + (
        f" and {replaced} of yours" if replaced else ""
    )

    def frame(k: int):
        stop = tour[k]
        info = seats["seats"][stop.key]
        drawing = pictures.Pictures(
            scene,
            info["seat"],
            info["movement"],
            facts=facts_of(info, heard),
            hrtf=args.hrtf,
            tool=settings_lines(DEMO, args, scene, report),
            credit=credit,
            heard=heard,
            played=played,
        )
        img = drawing.cover()
        tour_bar(img, tour, k, seats, pictures)
        return img

    frames, chapters = [], []
    for k, stop in enumerate(tour):
        info = seats["seats"][stop.key]
        pause = PAUSE_S if last_of_movement(tour, k) else 0.0
        frames.append((stop.key, stop.length + pause))
        numeral = MOVEMENTS[info["movement"]][0]
        chapters.append(f"{mmss(stop.at)} {numeral} · {TITLES[info['seat']][0]}")
    first = seats["seats"][tour[0].key]
    movements = len({seats["seats"][stop.key]["movement"] for stop in tour})
    places = len({seats["seats"][stop.key]["seat"] for stop in tour})
    thumbnail = pictures.Pictures(
        scene, first["seat"], first["movement"], credit=credit, heard=heard, played=played
    ).thumbnail(
        note=f"{movements} movement{'s' * (movements != 1)} · {places} seat"
        f"{'s' * (places != 1)} · {played}"
    )
    save(thumbnail, product(args, "_thumbnail.png"))
    pipeline.video(
        run_folder(args),
        frames,
        frame,
        sound,
        out=product(args, ".mp4"),
        no_mp4=args.no_mp4,
        chapters=chapters,
        shown=lambda k: seats["seats"][tour[k].key]["seat"],
    )


def last_of_movement(tour, k: int) -> bool:
    """Whether stop ``k`` ends its movement, so that the pause after it is its."""
    return k == len(tour) - 1 or not tour[k].joined_after


def tour_bar(img, tour, k: int, seats: dict, pictures) -> None:
    """A bar across the picture with a segment per stop, the current one bright, the
    movements' numerals over it."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    left, right, top, height = 80, 1840, 172, 22
    total = tour[-1].end
    fonts = [pictures.font("SemiBold", size) for size in (17, 14)]
    for j, stop in enumerate(tour):
        info = seats["seats"][stop.key]
        x0 = left + (right - left) * stop.at / total
        x1 = left + (right - left) * stop.end / total
        colour = pictures.LISTENER if j == k else (pictures.LINE if j < k else pictures.PANEL)
        draw.rounded_rectangle([x0, top, x1 - 3, top + height], 5, colour)
        label = f"{MOVEMENTS[info['movement']][0]} · {SHORT[info['seat']]}"
        # The label in the larger font that fits the segment, if one does.
        fits = [font for font in fonts if draw.textlength(label, font=font) < x1 - x0 - 8]
        if fits:
            fill = pictures.BG if j == k else pictures.MUTED
            draw.text(
                ((x0 + x1) / 2, top + height / 2), label, font=fits[0], fill=fill, anchor="mm"
            )


DEMO = Demo(
    name="orchestra",
    example="orchestra",
    title="Beethoven's 8th in the concert hall",
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step(
            "tracks",
            tracks,
            "download the 61 tracks of each movement (2.0 GB for the three)",
            tracks_done,
            tracks_download,
        ),
        SIMULATE,
        Step(
            "render",
            render,
            "each stop of the tour: a movement at a seat, in --format (binaural by default)",
        ),
        Step("video", video, "the run's video, its audio and the stills, named after the run"),
    ],
    options=[
        Option(
            "scene",
            "the example's scene",
            Path("scene.yaml"),
            "file",
            filter="Scene files (*.yaml *.yml)",
        ),
        Option(
            "results",
            "folder written by simulate, a folder in it for each output",
            Path("results"),
            "folder",
        ),
        Option(
            "zips",
            "a folder with movementN.zip downloaded before (or unpacked into movementN); "
            "the tracks of a movement not there come from the TU Berlin's server",
            Path("."),
            "folder",
            steps=("tracks",),
        ),
        *common_options(),
        input_option(),
        listeners_option(
            help="only these seats, comma separated, in the order each movement visits "
            "them (default: every one, in the tour's order; a --tour given keeps its own "
            "order, this option leaving out the others): each movement is heard at the "
            "seats of its tour among them, or at all of them when its tour has none of "
            "them; the simulation keeps every seat, so another choice needs no new one"
        ),
        *sources_options(
            help="only these players, by name (fl1, vl1_01) or by section (Strings takes "
            "Strings/Violins I to Strings/Double basses; Winds, Brass, Percussion), comma "
            "separated (default: all 61); the simulation keeps every player, so another "
            "pick needs no new one",
            steps=("render", "video"),
        ),
        Option(
            "movements",
            "only these movements, by number (1, 2, 4), comma separated, in this order "
            "(default: the tour's, 1,2,4)",
            None,
            steps=("tracks", "render", "video"),
        ),
        Option(
            "tour",
            "movement:seat,... (the seats of a movement take equal shares of it)",
            None,
            steps=("tracks", "render", "video"),
        ),
    ],
    simulations=simulations,
    problems=problems,
    plays=plays,
    listeners=lambda args: listeners_in(args.scene),
    sources=sources,
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
