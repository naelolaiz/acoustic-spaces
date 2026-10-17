# Translating

The editor, the command line and the messages of the library show their text in the
language [chosen as Getting started describes](getting-started.md#language). English is
the text in the code; each other language has a catalog, `src/acoustic_spaces/locale/<code>.po`
(gettext's format, which translation tools such as Poedit and Weblate read). Spanish
(`es.po`) and Catalan (`ca.po`) cover the editor, its plots and the examples' descriptions.

What stays in English in every language: scene files (their keys and values), command
and option names, the choices options take (`--early ray_tracing`), the files the
commands write (JSON, CSV, WAV and their column names), material ids and log files.
Scripts that read the command line's output keep working whatever the language.

## Marking text

Text a person reads goes through `acoustic_spaces.i18n`:

```python
from acoustic_spaces.i18n import N_, _, ngettext, pgettext

label = _("Air temperature")
message = _("Cannot open {path}: {reason}").format(path=path, reason=reason)
rooms = ngettext("{count} room", "{count} rooms", n).format(count=n)
band = pgettext("frequency band", "Band")  # where English uses one word for two things
LABELS = [N_("Open"), N_("Save")]  # module constants: marked here, _() where shown
```

- Fields have names (`{path}`, not `{}`), so a translation can put them in another order.
  An f-string inside `_()` cannot be translated: the catalog tool refuses it.
- Whole sentences, never pieces joined together: word order differs between languages.
- A comment starting with `Translators:` on the line just above goes into the catalog for
  the translator.
- `_` is the translation: a function that also names a variable `_` (`path, _ = ...`)
  cannot call it, and the catalog tool says so. Name the variable otherwise.
- The code never compares shown text: it keeps a value (a scene-file value, a level) and
  shows its translated label. A list of choices keeps the value as each entry's data
  (`combo.addItem(_(label), value)`, read with `currentData()`); the English label of
  a scene-file value is the value itself (`ray_tracing`), the Spanish one words
  (*trazado de rayos*), and the tips that name the choices use the same words.
- In the editor, numbers in a text go through `acoustic_spaces.gui.numbers`:
  `fmt(_("{time:.1f} s"), time=t)` writes them with the computer's decimal mark, and
  `parse_number` reads what a person typed in a table or field either way.

## The catalogs

```sh
python tools/catalog.py update   # collect the marked text into every catalog
python tools/catalog.py check    # what CI runs
```

`update` adds new text untranslated and drops text the code no longer has; text close to
some that is gone comes with the old translation marked `fuzzy`, to review. `check` fails
when a catalog is out of date, has text untranslated or fuzzy, or a translation whose
fields (`{name}`, `%s`) or menu shortcut (`&`) differ from the English. Both need only
Python's standard library. The example scenes' titles (`name`) and descriptions (the
comment at the top of `examples/*/scene.yaml`) are collected too, in the context
`example`; the editor shows them with `pgettext("example", text)`.

`ACOUSTIC_SPACES_LANG=pseudo` shows every translatable text as ⟦text⟧, so text without
brackets in the editor or the command line is text not yet marked. A test opens the
editor in it and fails on any menu, panel title, tab, button, form label, list entry or
tip that shows without brackets.

## Spanish

The Spanish is neutral, for readers in Spain and Latin America alike, and follows the
Spanish editions of the standards (UNE-EN ISO 3382 for room acoustic parameters,
UNE-EN ISO 354 for absorption, UNE-EN ISO 17497 for scattering). It translates terms of
the field as the field says them, not word for word.

Style:

- Actions (buttons, menu entries) in the infinitive: *Abrir*, *Guardar como…*, *Calcular*.
- Tooltips and messages describe; where they must address the reader they use impersonal
  forms (*se*, *hace falta*) rather than *tú* or *usted*.
- Words common to both sides of the Atlantic: *archivo* (not *fichero*), *equipo* for a
  computer (not *ordenador* or *computadora*).
- Sentence case and no full stop at the end of a label or tooltip, as in English.
- Quotation marks « ». Units and their symbols as in English (m², °C, kPa, dB, Hz), with
  a space before the unit. The editor writes numbers as the computer's regional settings
  do (0,85 in Spain, 0.85 in Mexico); the command line keeps the decimal point.
- Symbols of parameters stay as they are (T20, T30, EDT, C50, C80, D50, Ts, G, STI,
  STIPA, IACC, SPL) and acronyms take no plural *s* (*las RIR*).
- Scene-file values shown in the editor stay as written, in code format: `ray_tracing`.
- One shortcut (`&`) per menu entry, on a letter no other entry of that menu uses.

Glossary:

| English | Spanish | Notes |
|---|---|---|
| room (the space simulated) | recinto | as UNE-EN ISO 3382; kinds of rooms by their names: *sala de conciertos*, *estudio*, *iglesia* |
| box room | recinto rectangular | |
| scene | escena | |
| model, model file | modelo, archivo de modelo | |
| floor plan | planta | the architectural term |
| corner (of a plan) | vértice | |
| wall, floor, ceiling | pared, suelo, techo | |
| up axis | eje vertical | |
| audience area | zona de público | |
| audience seating | butacas | |
| leak (a gap rays escape through) | fuga | |
| closed (model) | cerrado | |
| surface, face, mesh | superficie, cara, malla | |
| source | fuente | point, line, surface source: *fuente puntual, lineal, superficial* |
| listener | receptor | the term of room acoustics, not *oyente* |
| sound power | potencia acústica | |
| directivity, directivity index | directividad, índice de directividad | |
| balloon (directivity) | globo de directividad | |
| loudspeaker, loudspeaker layout | altavoz, configuración de altavoces | 5.1, 7.1.4 as they are |
| array (of loudspeakers) | array | the usual word in Spanish sound engineering |
| gain, delay, equalizer | ganancia, retardo, ecualizador | |
| speech spectrum (male, female) | espectro de voz (masculina, femenina) | |
| absorption (coefficient) | absorción (coeficiente de absorción) | |
| scattering (coefficient) | dispersión (coeficiente de dispersión) | UNE-EN ISO 17497-1 |
| diffusion | difusión | not the same as scattering |
| sound reduction index R | índice de reducción sonora R | |
| partition, opening | elemento separador, abertura | |
| air absorption | absorción del aire | |
| run (a simulation) | calcular, cálculo | the button *Calcular*; a run that went: *cálculo* |
| early reflections | primeras reflexiones | |
| late reverberation, late tail | reverberación tardía, cola reverberante | |
| ray tracing (stochastic) | trazado de rayos (estocástico) | |
| image sources | fuentes imagen | the method: *método de las fuentes imagen* |
| diffuse rain | lluvia difusa | |
| secondary sources | fuentes secundarias | |
| edge diffraction | difracción en aristas | |
| wave solver | método ondulatorio | FDTD as it is |
| low frequencies | bajas frecuencias | |
| in pressure (coherent sum) | en presión | |
| seed (random) | semilla | |
| device (CPU, GPU) | dispositivo | CPU and GPU as they are, feminine: *la GPU* |
| impulse response | respuesta al impulso | RIR, BRIR as they are |
| echogram, decay curve | ecograma, curva de caída | |
| octave band, third-octave band | banda de octava, banda de tercio de octava | |
| sample rate | frecuencia de muestreo | |
| reverberation time | tiempo de reverberación | T20, T30 |
| early decay time | tiempo de caída inicial | EDT |
| clarity, definition | claridad, definición | C50, C80; D50 |
| centre time | tiempo central | Ts |
| sound strength | fuerza sonora | G |
| speech transmission index | índice de transmisión del habla | STI |
| sound pressure level | nivel de presión sonora | SPL |
| just noticeable difference | diferencia apenas perceptible | JND |
| background noise | ruido de fondo | |
| coverage map | mapa de cobertura | |
| binaural, Ambisonics, HRTF, SOFA, VBAP | as they are | *orden* for an Ambisonics order |
| spherical head | cabeza esférica | |
| head tracking | seguimiento de cabeza | |
| auralization | auralización | |
| convolution, deconvolution | convolución, deconvolución | |
| dry recording | grabación anecoica | |
| sweep (exponential sine) | barrido (senoidal exponencial) | |
| recording, playback | grabación, reproducción | |
| walkthrough | recorrido | |
| results, messages | resultados, mensajes | the editor's panels |
| command line | línea de comandos | |
| warning, error (message levels) | aviso, error | |

## Catalan

The Catalan catalog covers the same text as Spanish. Select **Català** under
**View > Language**, or set `ACOUSTIC_SPACES_LANG=ca`. System locale names such as
`ca_ES.UTF-8`, `ca-AD` and `ca_ES@valencia` also select it.

Style:

- Actions in the infinitive: *Obrir*, *Desar*, *Calcular*, *Cancel·lar*.
- Tooltips and messages use impersonal forms, sentence case and no final full stop
  on labels or tooltips, as in English.
- Use *fitxer*, *ordinador* and *per defecte*. Keep the product name Acoustic Spaces,
  file patterns, format names, units and parameter symbols (T20, C80, STI) as written.
- Use « » for quotation marks. Numbers follow the computer's regional settings;
  the command line keeps the decimal point.
- Each menu entry has one `&` shortcut, distinct from the others in that menu.
- Catalan uses two plural forms: singular for 1, plural for all other counts.

Glossary:

| English | Catalan |
|---|---|
| room, box room | recinte, recinte rectangular |
| scene, model file | escena, fitxer de model |
| floor plan, corner | planta, vèrtex |
| wall, floor, ceiling | paret, terra, sostre |
| source, listener | font, receptor |
| sound power, sound pressure level | potència acústica, nivell de pressió sonora |
| directivity, loudspeaker | directivitat, altaveu |
| absorption, scattering, diffusion | absorció, dispersió, difusió |
| sound reduction index | índex de reducció sonora |
| early reflections, late reverberation | primeres reflexions, reverberació tardana |
| ray tracing, image sources | traçat de raigs, fonts imatge |
| wave solver, edge diffraction | mètode ondulatori, difracció en arestes |
| impulse response, reverberation time | resposta impulsional, temps de reverberació |
| early decay time, decay curve | temps de decaïment inicial, corba de decaïment |
| clarity, definition, sound strength | claredat, definició, força sonora |
| speech transmission index | índex de transmissió de la parla |
| octave band, third-octave band | banda d'octava, banda de terç d'octava |
| sample rate, background noise | freqüència de mostreig, soroll de fons |
| coverage map, leak | mapa de cobertura, fuita |
| auralization, dry recording | auralització, enregistrament anecoic |
| playback, head tracking | reproducció, seguiment del cap |
| gain, delay, equalizer | guany, retard, equalitzador |
| results, messages, warning | resultats, missatges, avís |
