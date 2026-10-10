# Independent source-only publication workflow

This directory has a new Git history. Its main is a curated source snapshot
at upstream3f6ea95cb835b89ba50b0ef76d66d14b8037643e, not that upstream commit.
The original working repository and its ZIP/weights/history stay private and unchanged.

The only remote here is personal, pointing to
https://github.com/cout-he/sustaincluster-carbon-scheduling.git.
Do not fetch the old repository into this publication Git object store.
The first push is forbidden until the user explicitly approves final contents.

History consists of a new root baseline commit and the three approved chore,
code/tests and report commits. E1 logic/tests are copied byte-for-byte; publication
documentation and ignore rules are adapted to the source-only release.
Use explicit reviewed paths, never git add ., force-push, mirror or --all.
Before any later publication inspect all reachable objects and recheck remote refs.

Keep raw inputs/full outputs/weights and environments outside Git; all policy
changes require tests, named commits and experiment hashes. Main is not automatically
merged with research. Old private archives keep their original provenance; new Git
history does not retroactively identify their source. See ../../../README.md for
external data, environment and publication boundaries.
