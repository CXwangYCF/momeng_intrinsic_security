# LogiCAM-JSON reproduction audit

The first JSON experiment changed both output format and reasoning method. It
introduced a custom PL/FOL/NM representation, a rule enum, a 12-step cap,
candidate verdict fields, and a consistency gate. Those choices make the result
unsuitable for isolating the effect of JSON formatting.

Version 2 changes presentation only:

- loads the complete released `logicam/prompt.txt` for PL and NM;
- loads the complete released `logicam/prompt_fol.txt` for FOL;
- preserves every demonstration image and all example reasoning text;
- uses the released query field order: Image, Context, Question, Options;
- returns only `reasoning_steps` and `final_answer` through JSON Schema; each
  array item is one complete original-style step, capped at 16 to prevent the
  constrained decoder from repeating an open JSON string indefinitely;
- scores every parseable `final_answer` without a cross-field consistency gate;
- uses temperature 0, one request attempt, and 4,096 output tokens;
- records the selected prompt path and SHA256 in metadata and every result.

The v1 result remains in the output directory for provenance. Version 2 writes
to a new file and does not resume from or overwrite v1.
