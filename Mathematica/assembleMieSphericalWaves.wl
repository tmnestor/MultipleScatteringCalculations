#!/usr/bin/env wolframscript
(* Splice the verified archive definitions into the template to produce the
   single integrated source, MieSphericalWaves.wl.

   Keeping the two apart matters: the template is authored and reviewed, the
   definitions block is machine-emitted from the archive, and the assembled
   file is a build product.  Editing the assembled file by hand would silently
   fork it from the archive, which is the failure this whole chain exists to
   prevent.

   Run after emitVerifiedDefs.wl:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
       Mathematica/assembleMieSphericalWaves.wl
*)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
template = Import[dir <> "MieSphericalWaves_template.wl", "Text"];
block = Import["/tmp/verified_defs_block.wl", "Text"];

marker = "(*@INSERT_ARCHIVE_DEFS@*)";
If[! StringContainsQ[template, marker],
  Print["**** the template has no ", marker, " marker ****"]; Exit[1]];

out = StringReplace[template, marker -> block];
Export[dir <> "MieSphericalWaves.wl", out, "Text"];

Print["assembled MieSphericalWaves.wl"];
Print["  template ", StringLength[template], " chars"];
Print["  block    ", StringLength[block], " chars"];
Print["  total    ", StringLength[out], " chars"];
Print["  lines    ", Length[StringSplit[out, "\n"]]];

(* The assembled file must at least parse.  A syntax error here means the
   emitted block and the template disagree about something structural, and it
   is far cheaper to catch it now than inside a physics run. *)
expr = Quiet@Check[ToExpression[out, InputForm, Hold], $Failed];
Print["  parses: ", expr =!= $Failed];
