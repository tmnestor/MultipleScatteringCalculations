#!/usr/bin/env wolframscript
(* ==========================================================================
   THE PERMANENT STORE OF THE SCALAR CUBE MOMENTS

   A closed form is computed ONCE.  The scalar moments

       E[m; d1..dD; w1..wW] = < d_d1 .. d_dD r^m ,  x_w1 .. x_wW 1_V >,   V = [-Del/2, Del/2]^3,

   come out of CubeMomentCore.wl as Pell-reduced closed forms, and the expensive ones take minutes each.
   The engine's own cache (Mathematica/cache/*.mx) is a machine-specific binary, is not kept under version
   control, and is discarded whenever the engine file changes.  This file turns it into a permanent,
   readable record:

       CubeScalarMoments.m       every closed form, as  {m, {d..}, {w..}} -> expression in Del
                                 (InputForm text: readable, diffable, kept with the sources);
       cube_higher_moments.json  the same numbers at Del = 1 to 30 digits, for the independent numerical
                                 route (scripts/crosscheck_cube_moments_ball_shell.py).

   USE.
     wolframscript -file Mathematica/CubeMomentStore.wl          gather the cache into the store (merging
                                                                 with what the store already holds)
     Get["CubeMomentStore.wl"] after CubeMomentCore.wl, then loadMomentStore[]
                                                                 teach the engine every stored value, so
                                                                 that E$ returns it without recomputing

   A value already in the store is never overwritten by a different one: a disagreement between a new
   computation and the stored closed form stops the run and names the moment.
   ========================================================================== *)

$storeDir = DirectoryName[$InputFileName];
$storeFile = FileNameJoin[{$storeDir, "CubeScalarMoments.m"}];
$storeJson = FileNameJoin[{$storeDir, "cube_higher_moments.json"}];

readMomentStore[] := If[FileExistsQ[$storeFile], Association[Get[$storeFile]], <||>];

(* E$ of the engine, taught the stored values *)
loadMomentStore[] := Module[{st = readMomentStore[]},
  KeyValueMap[(E$[#1[[1]], #1[[2]], #1[[3]]] = #2) &, st];
  Length[st]];

parseKey[name_String] := Module[{parts = StringCases[name,
     "E_" ~~ m : (("-" | "") ~~ DigitCharacter ..) ~~ "_d" ~~ d : DigitCharacter ... ~~ "_w" ~~ w : DigitCharacter ... ~~ EndOfString :>
      {ToExpression[m], ToExpression /@ Characters[d], ToExpression /@ Characters[w]}]},
  If[parts === {}, Missing[], First[parts]]];

(* the value at Del = 1 to 30 digits.  A moment is real; a closed form written through complex
   intermediates (logarithms of negative numbers that cancel) is reported, and its real part exported. *)
numValue[key_, expr_] := Module[{v = Quiet[Block[{$MaxExtraPrecision = 1000}, N[expr /. Del -> 1, 30]]]},
  If[Head[v] === Complex,
   If[Abs[Im[v]] > 10^-20, Print["  NOT REAL: ", key, " = ", v]];
   Print["  written through complex intermediates: ", key];
   v = Re[v]];
  If[Abs[v] < 10^-25, 0, v]];

gatherMomentStore[] := Module[{st = readMomentStore[], files, added = 0, same = 0, key, val, num},
  files = FileNames["E_*.mx", FileNameJoin[{$storeDir, "cache"}]];
  Do[
   key = parseKey[FileBaseName[f]];
   If[! MissingQ[key],
    val = Import[f];
    If[KeyExistsQ[st, key],
     num = N[(st[key] - val) /. Del -> 37/29, 40];
     If[Abs[num] > 10^-30, Print["DISAGREEMENT with the store at ", key, ": ", num]; Abort[]];
     same++,
     st[key] = val; added++]],
   {f, files}];
  st = KeySort[st];
  Put[Normal[st], $storeFile];
  Export[$storeJson,
   <|"cube" -> "[-1/2, 1/2]^3", "moments" -> KeyValueMap[
       <|"m" -> #1[[1]], "d" -> #1[[2]], "w" -> #1[[3]], "value" -> numValue[#1, #2]|> &, st]|>, "JSON"];
  Print["moment store: ", Length[st], " closed forms (", added, " added, ", same, " already present and equal)"];
  Print["  ", $storeFile];
  Print["  ", $storeJson];
  Length[st]];

If[! ValueQ[$MomentCoreLoaded], gatherMomentStore[]];
