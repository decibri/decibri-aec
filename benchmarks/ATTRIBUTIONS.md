# Third-party data and models

This project's benchmarking uses third-party material that is not distributed
with it: the Microsoft AEC-Challenge dataset and the AECMOS scoring model. This
material is used under the terms below.

## Dataset: Microsoft AEC-Challenge

<https://github.com/microsoft/AEC-Challenge>

Provided under the original terms Microsoft received the source material under:

- Clean speech, LibriVox: public domain (https://librivox.org/pages/public-domain/)
- Clean speech, Edinburgh 56 speaker dataset: CC-BY-4.0 (https://datashare.is.ed.ac.uk/handle/10283/2791)
- Noise, AudioSet: CC-BY-4.0 (https://research.google.com/audioset/index.html)
- Noise, Freesound: CC0 (https://freesound.org/)
- Noise, DEMAND: CC-BY-SA-3.0 (https://zenodo.org/record/1227121)

## Scorer: AECMOS

The AECMOS reference implementation (from the AEC-Challenge repository) is MIT
licensed, copyright Microsoft Corporation. AECMOS is the objective metric used to
rank submissions in the Microsoft AEC Challenges; the paper is arXiv:2110.03010.

## Required citations

Publications using this dataset must cite the challenge papers; the AECMOS paper
covers the metric.

    @inproceedings{cutler2022AEC,
      title={ICASSP 2022 Acoustic Echo Cancellation Challenge},
      author={Cutler, Ross and Saabas, Ando and Parnamaa, Tanel and Purin, Marju
              and Gamper, Hannes and Braun, Sebastian and Sorensen, Karsten and
              Aichner, Robert},
      booktitle={ICASSP 2022}, year={2022}
    }

    @misc{cutler2023icassp,
      title={ICASSP 2023 Acoustic Echo Cancellation Challenge},
      author={Cutler, Ross and Saabas, Ando and Parnamaa, Tanel and Purin, Marju
              and Indenbom, Evgenii and Ristea, Nicolae-Catalin and Guzvin, Jegor
              and Gamper, Hannes and Braun, Sebastian and Aichner, Robert},
      year={2023}, eprint={2309.12553}, archivePrefix={arXiv}, primaryClass={eess.AS}
    }

    @misc{purin2021aecmos,
      title={AECMOS: A speech quality assessment metric for echo impairment.},
      author={Marju Purin, Sten Sootla, Mateja Sponza, Ando Saabas, and Ross Cutler},
      year={2021}, eprint={2110.03010v3}, archivePrefix={arXiv}, primaryClass={eess.AS}
    }
    