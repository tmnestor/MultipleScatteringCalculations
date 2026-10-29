#!/usr/bin/env wolframscript
(* ============================================================================================================
   runnb.wls: evaluate a notebook top to bottom in this kernel, save it with its outputs, and report them.

   usage:  wolframscript -file runnb.wls NOTEBOOK.nb --report=REPORT.json

   Run by scripts/build_notebooks.py in a scratch folder. NotebookEvaluate cannot be used here: run from a
   script kernel it sends Print to the terminal and inserts no results. So each Input cell is evaluated here, in
   order, and followed in the saved notebook by what it printed (Print cells, typeset), the messages it issued
   (Message cells, in the order issued) and its result (an Output cell, unless Null). No front end is needed.

   The report lists those output cells in order, as plain text with their style:
     {"status": "finished" | "exited" | "aborted", "exit_code": n or null, "version": $Version,
      "seconds": wall time, "cells": [{"style": "Print" | "Message" | "Output", "text": "..."}, ...]}
   "exited" means the script called Exit (nbExit in the notebook); the build judges the report, not this file.
   Exit codes: 0 when the report was written, 2 when the notebook could not be read or saved.
   ============================================================================================================ *)

(* the notebook evaluates in this kernel and context; the runner keeps its own names out of its way *)
runnb`die[what_String, fix_String] := (
  WriteString["stderr", "runnb: ERROR\n  What: " <> what <> "\n  Where: the command line\n  Should look like: " <>
    "wolframscript -file runnb.wls NOTEBOOK.nb --report=REPORT.json\n  Fix: " <> fix <> "\n"];
  Exit[2]);

runnb`args = Rest[$ScriptCommandLine];
runnb`nbPath = SelectFirst[runnb`args, ! StringStartsQ[#, "--"] &, None];
runnb`reportArg = SelectFirst[runnb`args, StringStartsQ[#, "--report="] &, None];
If[runnb`nbPath === None || ! FileExistsQ[runnb`nbPath], runnb`die["notebook not found: " <> ToString[runnb`nbPath], "pass an existing .nb"]];
If[runnb`reportArg === None, runnb`die["no --report=REPORT.json", "name the file the report is written to"]];
runnb`nbPath = ExpandFileName[runnb`nbPath];
runnb`reportPath = ExpandFileName[StringDrop[runnb`reportArg, 9]];
runnb`nb = Quiet[Get[runnb`nbPath]];
If[Head[runnb`nb] =!= Notebook, runnb`die["not a notebook: " <> runnb`nbPath, "pass a .nb written by wl2nb.wls"]];

(* ---------------------------------------------------------------------------------------------------------
   capture: Print is blocked to record its arguments; messages go to a file stream, which is read back (and
   restarted) before every Print and at the end of each cell, so the cells keep the order of issue
   --------------------------------------------------------------------------------------------------------- *)
runnb`msgFile = FileNameJoin[{$TemporaryDirectory, "runnb-messages-" <> ToString[$ProcessID] <> ".txt"}];
runnb`out = {};          (* the current cell's output cells, in order *)
runnb`report = {};       (* every output cell, as {style, text} *)
runnb`maxText = 4000;    (* report text per cell; the notebook keeps the whole output *)

runnb`emit[cell_Cell, text_String] := (AppendTo[runnb`out, cell];
   AppendTo[runnb`report, <|"style" -> cell[[2]], "text" -> StringTake[text, UpTo[runnb`maxText]]|>]);

runnb`openMessages[] := ($Messages = {OpenWrite[runnb`msgFile, PageWidth -> Infinity, CharacterEncoding -> "UTF-8"]});
runnb`drainMessages[] := Module[{text},
   Close /@ $Messages;
   text = Quiet[ReadString[runnb`msgFile]];
   If[StringQ[text] && StringTrim[text] =!= "",
    Do[runnb`emit[Cell[m, "Message", "MSG"], m],
     {m, StringTrim /@ StringSplit[StringTrim[text], RegularExpression["\\n(?=[A-Za-z$][\\w$`]*::[\\w$]+:)"]]}]];
   runnb`openMessages[]];

runnb`print[args___] := (runnb`drainMessages[];
   runnb`emit[Cell[BoxData[ToBoxes[Row[{args}]]], "Print"], ToString[Row[{args}]]]; Null);

runnb`line = 0;
runnb`status = "finished";

runnb`evalInput[boxes_] := Module[{held, result},
   runnb`out = {};
   runnb`line++;
   held = Quiet[ToExpression[boxes, StandardForm, HoldComplete]];
   If[Head[held] =!= HoldComplete,
    runnb`emit[Cell["runnb::parse: this cell could not be parsed", "Message", "MSG"],
     "runnb::parse: this cell could not be parsed"];
    Return[runnb`out, Module]];
   result = Block[{$Messages}, runnb`openMessages[];
     With[{r = CheckAbort[Block[{Print = runnb`print}, ReleaseHold[held]], runnb`abortedQ = True; $Aborted]},
      runnb`drainMessages[]; Close /@ $Messages; r]];
   If[TrueQ[runnb`abortedQ], runnb`status = If[TrueQ[$nbStopped], "exited", "aborted"]];
   If[result =!= Null && ! TrueQ[runnb`abortedQ],
    runnb`emit[Cell[BoxData[ToBoxes[result]], "Output", CellLabel -> "Out[" <> ToString[runnb`line] <> "]="],
     ToString[Short[result, 40]]]];
   runnb`out];

(* walk the cells in order; after an exit or abort the rest are kept as they are, unevaluated *)
runnb`walk[cells_List] := Join @@ (runnb`walk1 /@ cells);
runnb`walk1[Cell[CellGroupData[cs_List, o___], r___]] := {Cell[CellGroupData[runnb`walk[cs], o], r]};
runnb`walk1[c : Cell[b_, "Input", r___]] := If[runnb`status =!= "finished", {c},
   With[{outs = runnb`evalInput[b]},
    Join[{Cell[b, "Input", r, CellLabel -> "In[" <> ToString[runnb`line] <> "]:="]}, outs]]];
runnb`walk1[c_] := {c};

runnb`t0 = AbsoluteTime[];
runnb`cells = runnb`walk[First[runnb`nb]];
runnb`seconds = Round[AbsoluteTime[] - runnb`t0, 0.1];
Quiet[DeleteFile[runnb`msgFile]];

If[Export[runnb`nbPath, Notebook[runnb`cells, Sequence @@ Rest[runnb`nb]], "NB"] === $Failed,
  runnb`die["could not save " <> runnb`nbPath, "check the folder is writable"]];
Export[runnb`reportPath, <|"status" -> runnb`status,
   "exit_code" -> If[runnb`status === "exited", $nbExitCode, Null], "version" -> $Version,
   "seconds" -> runnb`seconds, "cells" -> runnb`report|>, "RawJSON"];
Print["runnb: ", runnb`status, " in ", runnb`seconds, " s, ", Length[runnb`report], " output cells"];
Exit[0];
