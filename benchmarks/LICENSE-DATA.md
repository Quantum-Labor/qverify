# Dataset attribution and licensing

QVerify benchmark runs ship with the following third-party datasets.
Each dataset retains its upstream license; the harness that consumes
them (the `qverify.eval` module) is Apache 2.0.

## ProofWriter

- **License:** Apache 2.0
- **Source:** https://allenai.org/data/proofwriter
- **Citation:** Tafjord, O., Dalvi, B., & Clark, P. (2021). ProofWriter:
  Generating Implications, Proofs, and Abductive Statements over Natural
  Language. *Findings of ACL 2021*.

## RuleTaker

- **License:** CC BY 4.0
- **Source:** https://allenai.org/data/ruletaker
- **Citation:** Clark, P., Tafjord, O., & Richardson, K. (2020).
  Transformers as Soft Reasoners over Language. *IJCAI 2020*.

## Note about FOLIO

FOLIO is intentionally not included in v0.1 because its CC BY-SA 4.0
license requires derivative benchmark reports to also be released under
CC BY-SA, which conflicts with this project's Apache 2.0 default. The
constraint applies to any published artifact (JSON reports, PNG charts,
README tables) that incorporates FOLIO outputs. Including it would force
the entire benchmark report into CC BY-SA, including the parts derived
from ProofWriter (Apache 2.0) and RuleTaker (CC BY 4.0). This may be
revisited in a future release if a separate publishing path is set up
for FOLIO results.

`qverify.eval.datasets.load_folio` raises `NotImplementedError` to make
the exclusion explicit at the call site.
