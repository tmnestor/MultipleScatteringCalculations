#!/usr/bin/env wolframscript
(* ============================================================================================================
   wl2nb.wls: convert a Wolfram Language script (.wl/.wls/.m) into an unevaluated notebook beside it.

   usage:  wolframscript -file wl2nb.wls SCRIPT.wl --spec=SPEC.json [--out=PATH.nb] [--force]

   Run by scripts/build_notebooks.py, which writes SPEC.json from the script's entry in notebooks.yml:
     {"intro": "text under the title" or null, "extra_cells": ["code", ...], "args": ["--flag", ...]}

   WHAT IT DOES
   - Top-level comments that start a line become prose: a banner comment (one with a rule line of = or -)
     becomes a Section heading plus Text; any other comment becomes Text; the file's leading banner becomes
     the Subtitle under a Title named after the file. Package-format markers (* ::Section:: *) are honoured.
   - Code between them becomes Input cells, split at blank lines (never inside an expression).
   - Get[...] calls stay as they are: libraries and data are loaded from the files beside the notebook, so the
     notebook runs only from this folder, and a library is never copied into a notebook where it could go stale.
   - Script-only symbols are rewritten to notebook stand-ins defined in a "Notebook set-up" cell:
       $InputFileName -> $nbInputFileName,  $ScriptCommandLine -> $nbScriptCommandLine (with the spec's
       arguments),  Exit[...] / Quit[...] -> nbExit[...] (records the exit code and stops the evaluation).
   - The spec's extra cells are appended after the script, under their own heading.
   - Parity check: the code in the cells must parse to exactly the expressions of the rewritten source.

   Exit codes: 0 success, 2 failure (with a diagnostic on stderr).
   ============================================================================================================ *)

$feUsed = False;

die[what_String, where_String, should_String, fix_String] := (
  WriteString["stderr", "wl2nb: ERROR\n  What: " <> what <> "\n  Where: " <> where <> "\n  Should look like: " <>
    should <> "\n  Fix: " <> fix <> "\n"];
  If[$feUsed, Quiet[Developer`UninstallFrontEnd[]]];
  Exit[2]);

(* ---------------------------------------------------------------------------------------------------------
   command line
   --------------------------------------------------------------------------------------------------------- *)
$usage = "wolframscript -file wl2nb.wls SCRIPT.wl --spec=SPEC.json [--out=PATH.nb] [--force]";
$args = Rest[$ScriptCommandLine];
$positional = Select[$args, ! StringStartsQ[#, "--"] &];
$force = MemberQ[$args, "--force"];
$outArg = SelectFirst[$args, StringStartsQ[#, "--out="] &, None];
$specArg = SelectFirst[$args, StringStartsQ[#, "--spec="] &, None];
With[{bad = Select[$args, StringStartsQ[#, "--"] && ! (StringStartsQ[#, "--out="] ||
        StringStartsQ[#, "--spec="] || # == "--force") &]},
  If[bad =!= {}, die["unknown option(s): " <> StringRiffle[bad, ", "], "the command line", $usage,
     "use only --spec=, --out= and --force"]]];
If[Length[$positional] != 1,
  die["expected exactly one script path, got " <> ToString[Length[$positional]], "the command line", $usage,
   "pass the path of one .wl/.wls/.m file"]];
If[$specArg === None, die["no --spec=SPEC.json", "the command line", $usage,
   "build notebooks with scripts/build_notebooks.py, which writes the spec from notebooks.yml"]];

$main = ExpandFileName[First[$positional]];
If[! FileExistsQ[$main], die["script not found: " <> $main, "the command line", $usage, "check the path"]];
If[! MemberQ[{"wl", "wls", "m"}, ToLowerCase[FileExtension[$main]]],
  die["not a Wolfram Language script: " <> $main, "the command line", "a file ending .wl, .wls or .m",
   "pass the script itself, not a notebook or data file"]];
$out = If[$outArg === None, FileNameJoin[{DirectoryName[$main], FileBaseName[$main] <> ".nb"}],
   ExpandFileName[StringDrop[$outArg, 6]]];
If[FileExistsQ[$out] && ! $force,
  die["output exists: " <> $out, "the command line", "a path with no notebook yet, or --force",
   "add --force to overwrite it, or choose another path with --out=PATH.nb"]];

$spec = Quiet[Import[ExpandFileName[StringDrop[$specArg, 7]], "RawJSON"]];
If[! AssociationQ[$spec] || Sort[Keys[$spec]] =!= {"args", "extra_cells", "intro"} ||
   ! (StringQ[$spec["intro"]] || $spec["intro"] === Null) ||
   ! (ListQ[$spec["extra_cells"]] && AllTrue[$spec["extra_cells"], StringQ]) ||
   ! (ListQ[$spec["args"]] && AllTrue[$spec["args"], StringQ]),
  die["the spec is missing, unreadable or malformed: " <> StringDrop[$specArg, 7], "the --spec= file",
   "{\"intro\": null, \"extra_cells\": [], \"args\": []}",
   "let scripts/build_notebooks.py write the spec; check the entry in notebooks.yml"]];

(* ---------------------------------------------------------------------------------------------------------
   scanning: split a file into top-level prose comments and code, with a mask of the code in which the
   contents of strings and comments are blanked (so token searches never match inside them)
   --------------------------------------------------------------------------------------------------------- *)
commentEnd[c_List, i0_Integer, n_Integer, file_String] := Module[{d = 0, k = i0},
  While[k < n,
   Which[
    c[[k]] == "(" && c[[k + 1]] == "*", d++; k += 2,
    c[[k]] == "*" && c[[k + 1]] == ")", d--; If[d == 0, Return[k + 1, Module]]; k += 2,
    True, k++]];
  die["unterminated comment", file <> " line " <> ToString[1 + Count[c[[;; i0]], "\n"]],
   "every (* closed by a matching *)", "close the comment in the source script"]];

scan[s_String, file_String] := Module[{c = Characters[s], m, n, i = 1, depth = 0, lineBlank = True,
   segs = {}, codeStart = 1, j, k, lineOf, flush},
  n = Length[c]; m = c;
  lineOf[p_] := 1 + Count[c[[;; p - 1]], "\n"];
  (* "col" is the 0-based column where the segment starts, so a comment's first line can be dedented with the rest *)
  flush[a_, b_, type_] := If[b >= a, AppendTo[segs, <|"type" -> type, "text" -> StringJoin[c[[a ;; b]]],
       "mask" -> StringJoin[m[[a ;; b]]], "line" -> lineOf[a],
       "col" -> a - 1 - (Last[Flatten[{0, Position[c[[;; a - 1]], "\n"]}]])|>]];
  While[i <= n,
   Which[
    c[[i]] == "\"",
    j = i + 1;
    While[j <= n && c[[j]] != "\"", If[c[[j]] == "\\", j += 2, j++]];
    Do[If[m[[k]] != "\n", m[[k]] = " "], {k, i + 1, Min[j - 1, n]}];
    lineBlank = False; i = j + 1,
    c[[i]] == "(" && i < n && c[[i + 1]] == "*",
    j = commentEnd[c, i, n, file];
    k = j + 1;
    While[k <= n && (c[[k]] == " " || c[[k]] == "\t"), k++];
    If[depth == 0 && lineBlank && (k > n || c[[k]] == "\n"),
     flush[codeStart, i - 1, "code"]; flush[i, j, "prose"];
     codeStart = k + 1; i = k + 1; lineBlank = True,
     Do[If[m[[k]] != "\n", m[[k]] = " "], {k, i, j}];
     lineBlank = False; i = j + 1],
    MemberQ[{"(", "[", "{"}, c[[i]]], depth++; lineBlank = False; i++,
    MemberQ[{")", "]", "}"}, c[[i]]], depth--; lineBlank = False; i++,
    c[[i]] == "\n", lineBlank = True; i++,
    True, If[c[[i]] != " " && c[[i]] != "\t", lineBlank = False]; i++]];
  flush[codeStart, n, "code"];
  segs];

(* a parse failure is a bare $Failed element; code that merely mentions $Failed parses to HoldComplete[...] *)
parseHeld[s_String] := Quiet[ImportString[s, {"Package", "HeldExpressions"}]];
parsedQ[r_] := ListQ[r] && AllTrue[r, MatchQ[#, _HoldComplete] &];
completeQ[s_String] := parsedQ[parseHeld[s]];

(* a prose comment that falls inside an unfinished expression is code after all: merge it back *)
mergeIncomplete[segs_List] := Module[{out = {}, acc = None},
  Do[
   Which[
    acc === None && seg["type"] == "code", acc = seg,
    acc === None, AppendTo[out, seg],
    completeQ[acc["text"]], AppendTo[out, acc]; acc = If[seg["type"] == "code", seg, AppendTo[out, seg]; None],
    True, acc = <|acc, "text" -> acc["text"] <> "\n" <> seg["text"],
       "mask" -> acc["mask"] <> "\n" <> If[seg["type"] == "code", seg["mask"],
          StringReplace[seg["text"], Except["\n"] -> " "]]|>],
   {seg, segs}];
  If[acc =!= None, AppendTo[out, acc]];
  out];

(* ---------------------------------------------------------------------------------------------------------
   rewriting a code segment: script-only symbols to notebook stand-ins
   --------------------------------------------------------------------------------------------------------- *)
$idChar = "[A-Za-z0-9$`]";
tokenEdits[mask_String] := Join[
  {#, "$nbInputFileName"} & /@ StringPosition[mask, RegularExpression["(?<!" <> $idChar <> ")\\$InputFileName(?!" <> $idChar <> ")"]],
  {#, "$nbScriptCommandLine"} & /@ StringPosition[mask, RegularExpression["(?<!" <> $idChar <> ")\\$ScriptCommandLine(?!" <> $idChar <> ")"]],
  {#, "nbExit"} & /@ StringPosition[mask, RegularExpression["(?<!" <> $idChar <> ")(Exit|Quit)(?=\\s*\\[)"]]];

applyEdits[text_String, tokEd_List] := If[tokEd === {}, text, StringReplacePart[text, tokEd[[All, 2]], tokEd[[All, 1]]]];

(* ---------------------------------------------------------------------------------------------------------
   prose comments to cells
   --------------------------------------------------------------------------------------------------------- *)
indentOf[l_String] := StringLength[l] - StringLength[StringTrim[l, StartOfString ~~ (" " | "\t") ..]];
ruleQ[l_String] := StringMatchQ[l, RegularExpression["\\s*[=\\-~#*]{8,}\\s*"]];
inlineHeading[l_String] := StringCases[l, RegularExpression["^\\s*[=\\-]{3,}\\s+(.+?)\\s+[=\\-]{3,}\\s*$"] :> "$1"];
bulletQ[l_String] := StringMatchQ[l, RegularExpression["\\s*[-*\[Bullet]]\\s+.*"]];

paragraphCells[lines_List] := Module[{base, cells = {}, cur = {}, curKind = None, kind, emit},
  emit[] := (If[cur =!= {}, AppendTo[cells, Switch[curKind,
        "pre", Cell[StringRiffle[cur, "\n"], "Program"],
        "item", Cell[StringRiffle[StringTrim /@ cur, " "], "Item"],
        _, Cell[StringRiffle[StringTrim /@ cur, " "], "Text"]]]]; cur = {});
  If[Length[lines] >= 2 && AnyTrue[lines, StringMatchQ[#, RegularExpression[".*\\S {3,}\\S.*"]] &],
   Return[{Cell[StringRiffle[lines, "\n"], "Program"]}, Module]];
  base = indentOf[First[lines]];
  Do[
   kind = Which[bulletQ[l], "item-start", indentOf[l] >= base + 2 && curKind =!= "item", "pre",
     curKind === "item" && indentOf[l] > base, "item", True, "text"];
   Switch[kind,
    "item-start", emit[]; curKind = "item"; AppendTo[cur, StringReplace[l, RegularExpression["^\\s*[-*\[Bullet]]\\s+"] -> ""]],
    _, If[kind =!= curKind, emit[]; curKind = kind]; AppendTo[cur, If[kind == "pre", StringDrop[l, base], l]]],
   {l, lines}];
  emit[];
  cells];

textCells[lines0_List] := Module[{lines = lines0, ind, paras},
  ind = Min[indentOf /@ Select[lines, StringTrim[#] =!= "" &] /. {} -> {0}];
  lines = If[StringLength[#] >= ind, StringDrop[#, ind], StringTrim[#]] & /@ lines;
  paras = Select[SplitBy[lines, StringTrim[#] === "" &], StringTrim[First[#]] =!= "" &];
  Join @@ (paragraphCells /@ paras)];

(* a banner's heading: the first sentence when one ends inside the first line, or when the first line runs on
   mid-sentence; otherwise the first line ("[1] the sphere: its series" followed by a new sentence) *)
headingSplit[lines_List] := Module[{para, rest, l1, joined, bounds, inLine1},
  para = TakeWhile[lines, StringTrim[#] =!= "" &];
  rest = Drop[lines, Length[para]];
  l1 = StringTrim[First[para]];
  joined = StringRiffle[StringTrim /@ para, " "];
  bounds = Select[StringPosition[joined, RegularExpression["(?<=[.:!?])\\s+(?=[A-Z(\\[])"]],
    ! StringMatchQ[StringTake[joined, #[[1]] - 1], RegularExpression["\\A\\s*(\\d+\\.|\\[\\d+\\])"]] &];
  inLine1 = bounds =!= {} && bounds[[1, 1]] <= StringLength[l1];
  Which[
   inLine1 || (Length[para] > 1 && ! StringMatchQ[l1, RegularExpression[".*[.:!?]"]] &&
       ! StringMatchQ[StringTrim[para[[2]]], RegularExpression["[A-Z\\[(].*"]]),
   If[bounds === {}, {joined, rest},
    {StringTake[joined, bounds[[1, 1]] - 1], Join[{StringDrop[joined, bounds[[1, 2]]]}, rest]}],
   True, {l1, Join[Rest[para], rest]}]];

(* {cells, pendingStyle}; headingStyle is the style a banner's heading takes *)
proseCells[text_String, col_Integer, pending_, headingStyle_String] := Module[{inner, lines, heads, marker, hs},
  inner = StringRepeat[" ", col + 2] <> StringTake[text, {3, -3}];
  marker = StringCases[StringTrim[inner], RegularExpression["\\A::(\\w+)::\\z"] :> "$1"];
  Which[
   marker =!= {}, {{}, If[First[marker] == "Package", None, First[marker]]},
   pending =!= None, {{If[MemberQ[{"Input", "Code", "Program"}, pending], Cell[StringTrim[inner], "Program"],
       Cell[StringTrim[inner], pending]]}, None},
   True,
   lines = StringSplit[inner, "\n", All];
   heads = Join @@ (inlineHeading /@ lines);
   If[AnyTrue[lines, ruleQ] || heads =!= {},
    lines = Select[lines, ! ruleQ[#] && inlineHeading[#] === {} &];
    lines = Drop[lines, LengthWhile[lines, StringTrim[#] === "" &]];
    If[heads =!= {},
     {Join[{Cell[First[heads], headingStyle]}, textCells[lines]], None},
     If[lines === {}, {{}, None},
      hs = headingSplit[lines];
      {Join[{Cell[First[hs], headingStyle]}, textCells[Last[hs]]], None}]],
    {textCells[lines], None}]]];

(* ---------------------------------------------------------------------------------------------------------
   code to cells
   --------------------------------------------------------------------------------------------------------- *)
trimCode[s_String] := Module[{ls = StringSplit[s, "\n", All]},
  StringReplace[StringRiffle[Drop[ls, LengthWhile[ls, StringTrim[#] === "" &]], "\n"], Whitespace ~~ EndOfString -> ""]];

chunks[code_String] := Module[{parts = StringSplit[code, RegularExpression["\\n[ \\t]*\\n"]], out = {}, acc = ""},
  Do[acc = If[acc === "", p, acc <> "\n\n" <> p];
   If[completeQ[acc], If[StringTrim[acc] =!= "", AppendTo[out, trimCode[acc]]]; acc = ""], {p, parts}];
  If[StringTrim[acc] =!= "", AppendTo[out, trimCode[acc]]];
  out];

toBoxes[s_String] := ($feUsed = True;
   BoxData[Replace[UsingFrontEnd[FrontEndExecute[FrontEnd`ReparseBoxStructurePacket[s]]], l_List :> RowBox[l]]]);

(* ---------------------------------------------------------------------------------------------------------
   converting the script
   --------------------------------------------------------------------------------------------------------- *)
$argLines = {};        (* "line N" where $ScriptCommandLine is read *)
$exitUsed = False;

convertFile[file_String] := Module[{s, segs, cells = {}, pending = None, codeStrings = {}, rewrittenAll = {},
   te, pc, code, first = True, headingStyle, origHeld, cellHeld},
  s = Quiet[ReadString[file]];
  If[! StringQ[s], die["cannot read " <> file, file, "a readable UTF-8 text file", "check the file's permissions"]];
  s = StringReplace[s, StartOfString ~~ "#!" ~~ Except["\n"] ... -> ""];
  segs = mergeIncomplete[scan[s, file]];
  Do[
   If[seg["type"] == "code",
    te = tokenEdits[seg["mask"]];
    Do[AppendTo[$argLines, "line " <> ToString[seg["line"] + StringCount[StringTake[seg["text"], t[[1, 1]] - 1], "\n"]]],
     {t, Select[te, #[[2]] == "$nbScriptCommandLine" &]}];
    $argLines = DeleteDuplicates[$argLines];
    If[AnyTrue[te, #[[2]] == "nbExit" &], $exitUsed = True];
    code = applyEdits[seg["text"], te];
    AppendTo[rewrittenAll, code];
    Do[AppendTo[codeStrings, ch]; AppendTo[cells, Cell[toBoxes[ch], "Input"]], {ch, chunks[code]}];
    If[StringTrim[seg["text"]] =!= "", first = False],
    (* prose *)
    AppendTo[rewrittenAll, seg["text"]];
    headingStyle = If[first, "Subtitle", "Section"];
    first = False;
    {pc, pending} = proseCells[seg["text"], seg["col"], pending, headingStyle];
    cells = Join[cells, pc]],
   {seg, segs}];
  (* parity: the cells' code parses to exactly the expressions of the rewritten source *)
  origHeld = parseHeld[StringRiffle[rewrittenAll, "\n"]];
  cellHeld = Join @@ (parseHeld /@ codeStrings);
  If[! parsedQ[origHeld] || origHeld =!= cellHeld,
   die["parity check failed: the notebook's code does not reproduce the script's " <>
     ToString[Length[origHeld]] <> " top-level expressions (cells give " <> ToString[Length[cellHeld]] <> ")",
    file, "the source parses cleanly under wolframscript, and every top-level comment sits between statements",
    "run the script once to confirm it parses; if it does, report this file as a converter bug"]];
  cells];

(* ---------------------------------------------------------------------------------------------------------
   assembly
   --------------------------------------------------------------------------------------------------------- *)
mainCells = convertFile[$main];
{headCells, bodyCells} = Module[{k = LengthWhile[mainCells, MemberQ[{"Subtitle", "Text", "Program", "Item"}, #[[2]]] &]},
   If[k > 0 && mainCells[[1, 2]] == "Subtitle", TakeDrop[mainCells, k], {{}, mainCells}]];

setupCode = StringRiffle[{
    "(* stand-ins for the script environment; this notebook is generated from " <> FileNameTake[$main] <> " *)",
    "$nbInputFileName = Quiet[Check[NotebookFileName[EvaluationNotebook[]], FileNameJoin[{Directory[], " <>
     ToString[FileNameTake[$out], InputForm] <> "}]]];",
    "$nbScriptCommandLine = Join[{$nbInputFileName}, " <> ToString[$spec["args"], InputForm] <> "];",
    "(* Exit/Quit: Abort[] alone stops only the current cell, so after an exit $Pre aborts every later cell,",
    "   except one that mentions $nbStopped (this cell), which resets it *)",
    "$nbStopped = False; $nbExitCode = None;",
    "nbExit[code_ : 0] := ($nbStopped = True; $nbExitCode = code; Print[\"[the script exits here with code \", code,",
    "    \"; later cells are skipped until this set-up cell is evaluated again]\"]; Abort[]);",
    "$Pre = Function[in, If[TrueQ[$nbStopped] && FreeQ[Hold[in], HoldPattern[$nbStopped]], Abort[], in], HoldAll];"},
   "\n"];
setupText = "Generated from " <> FileNameTake[$main] <> " (the script is the source; edit it, not this notebook). " <>
   "To rerun, open this notebook from the folder holding the script and the files it loads, and use " <>
   "Evaluation \[RightGuillemet] Evaluate Notebook. The cell below stands in for the script environment: " <>
   "$InputFileName became $nbInputFileName, $ScriptCommandLine became $nbScriptCommandLine and Exit/Quit became " <>
   "nbExit, which skips every later cell instead of ending the session (evaluate this cell again to reset)." <>
   Which[
    $spec["args"] =!= {}, " This notebook runs the script with the arguments " <> StringRiffle[$spec["args"], " "] <> ".",
    $argLines =!= {}, " The script reads command-line arguments (" <> StringRiffle[$argLines, ", "] <>
     "); this notebook passes none, so it takes its defaults.",
    True, ""];
setupGroup = Cell[CellGroupData[{Cell["Notebook set-up", "Section"], Cell[setupText, "Text"],
     Cell[toBoxes[setupCode], "Input"]}, Open]];

introCells = If[$spec["intro"] === Null, {}, {Cell[$spec["intro"], "Text"]}];
extraGroup = If[$spec["extra_cells"] === {}, Nothing,
   Cell[CellGroupData[Join[{Cell["Added to the notebook", "Section"],
       Cell["Not part of " <> FileNameTake[$main] <> ": run here because the script runs it only when executed " <>
         "as a script.", "Text"]},
      Cell[toBoxes[#], "Input"] & /@ $spec["extra_cells"]], Open]]];

nb = Notebook[Join[{Cell[FileBaseName[$main], "Title"]}, introCells, headCells, {setupGroup}, bodyCells, {extraGroup}]];
If[$feUsed, Quiet[Developer`UninstallFrontEnd[]]];
If[Export[$out, nb, "NB"] === $Failed,
  die["could not write " <> $out, "the output path", "a writable directory", "choose another path with --out="]];

Print["wrote ", FileNameTake[$out]];
Print["  cells: ", Count[nb, Cell[_, "Input", ___], Infinity], " input, ",
  Count[nb, Cell[_, "Text" | "Item" | "Program", ___], Infinity], " prose, ",
  Count[nb, Cell[_, "Section" | "Subsection" | "Subsubsection", ___], Infinity], " headings"];
If[$argLines =!= {}, Print["  reads command-line arguments: ", StringRiffle[$argLines, ", "]]];
If[$exitUsed, Print["  Exit/Quit rewritten to nbExit"]];
Print["  parity: the cells reproduce the script's top-level expressions"];
Exit[0];
