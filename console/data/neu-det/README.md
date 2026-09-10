# data/neu-det/test/images -- 18 of 180

A slice of the held-out NEU-DET test split, not the dataset. Three frames per
defect family, 18 in total, so the console has something to score with no upload
and every class is one click away.

**Which three, and why those.** For each family the 30 held-out frames are sorted
by filename and the first, middle and last are taken -- indices 0, 15 and 29. A
deterministic rule rather than a curated pick: choosing the frames the model does
well on would make the demonstration a lie, and choosing at random would make it
unreproducible. Reproduce the selection with:

    for each family: frames = sorted(data/neu-det/test/images/<family>_*.jpg)
    take frames[0], frames[len//2], frames[-1]

**Provenance.** These are real held-out images. The 180-frame test split was
never trained on and never tuned on -- `src/prepare_data.py` carves the 180-frame
validation set out of the upstream train split precisely so the test split can be
reported once. So every detection, severity and latency the console prints over
these frames is a measurement on unseen data.

**The thing to know before reading a defect rate.** NEU-DET contains no
defect-free image, in any split: all 1,800 frames carry at least one labelled
defect. A batch drawn from here therefore has a defect rate near 100% by
construction, and that figure says nothing about how often this detector would
flag sound steel. The console prints that caveat next to every rate it computes
from these frames.

No label files are shipped. The console reads the ground-truth family from the
filename, which is all a sample picker needs; nothing here computes accuracy at
runtime.
