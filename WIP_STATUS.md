# Work in progress: a wall through a plane of cell centres (wave solver)

Not for merging. Saved as it stands on 2026-11-11.

## What is here
The third round of the fix for a wall that lies exactly in a plane of cell centres (it reads 2.5 to
7 dB low at grazing today, because hits nearer than 1e-9 of the model's size are dropped): one tie
rule for faces through cell centres, and a rule for a source or listener that lies on a face. The
wave tests pass with it.

## What is missing
An independent check found two cases this round does not hold:
1. A source or listener exactly on an edge or a vertex whose faces lie in planes of cell centres
   misses the centre on that edge (likely cause: in `crossing()`, the case of both ends on the
   plane and on different sides takes the meeting point by the foot's rule where the rule towards
   the other end is needed).
2. Results are no longer identical to the bit where two faces of different materials coincide
   (a building on a ground plate: 181 boundary cells, differences of 8.5e-6). They must be, since
   no such scene is wrong today.

## Next
Repair both, sweep points on every kind of feature (face, edge, vertex, coincident faces)
systematically, have it checked independently, measure the set-up time on the 19-point grid, then
rebase onto the stack, add docs and changelog, and open the pull request.
