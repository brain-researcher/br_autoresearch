# Provider mask construction question — draft only

Please clarify the construction provenance of the released native T1w
`laion-general` masks for sub-01, sub-03, sub-05, sub-06, and sub-07.

Our planned adapted neural-code-conversion analysis must exclude regular shared
`tau` variant 0 test responses and all 371 OOD image responses and derivatives
from mask or voxel selection. Public documentation states that `laion-general`
was delineated from main-experiment GLMsingle R², but does not identify the
selecting trial pool.

Could you identify the exact mask revision and source R² maps, the included
image/trial/session IDs, any upstream aggregation or selection contributing
to those maps, the delineation rule, and the surface-to-native-T1w conversion?
Please state explicitly whether tau variant 0 test responses or any OOD
responses contributed. If they did, is a mask defined solely from independent
localizers or an eligible training-only response pool available?

We need provenance metadata only, not neural values or outcome summaries.
This is an unsent draft; no contact has been made on the scientist's behalf.
