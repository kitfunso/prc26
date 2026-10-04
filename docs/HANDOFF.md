# Hand-off, 2026-09-22

> **Current campaign checkpoint, 4 October 2026, 21:46 London.** The full January/July
> clean reference is now measured and checked. All three inner complete-file tables
> pass native lineage, scopes, raw/floor arithmetic and identical original mixture B.
> Command: `python analysis/champion_assembly_audit.py --meta-group 4 8 11`.
> The last table covers 528,801 rows: 524,031 Lane A and 4,770 Lane B, in complete
> April/August/November files. Its output SHA is
> 5dda261c746f2d82b1d1a930f7a1b7486a74ad227c23e58416c699fff509174c.
> `python analysis/champion_evaluation_audit.py --months 1-7` passes actual cold bytes,
> complete-file coverage, preserved labels, inner-table lineage, training-only NNLS
> inputs and weight reproduction, 300-tree residual replay, recording-context
> reproduction, stopping/history scopes, native final-B artifacts, recording replay
> and B parity. Full proof: 1-7/evaluation-audit.json under the clean-champion run.
> Whole-file RMSE: 309.1881 control to 304.2788 clean recipe, gain 4.9092 seconds;
> paired day interval [3.3573, 6.7721], five-day [3.6247, 6.5968]. January and July
> improve separately; all ten airport slices and the hot-weather slice improve.
> Gain remains after removing the largest day or five-day episode. AOBT-copy/body
> diagnostic slices lose; positive/negative tails gain. These label-based strata are
> descriptive and cannot be ranking features. Final B is identical in both arms.
> This compares the clean recipe with its scoped CatBoost control, not uploaded v53.
> No new board gain or upload. Evaluation SHA:
> f929e2fa89ab7fff2d965812657822784cd1d3b28aa0a7feb9bda7281e1c80d9.
> Saved component wall cost: 57,554.9595 seconds; GPU-family proxy: 44,191.8195
> seconds (12.28 hours). First-block reference/evaluation stages are checked;
> remaining confirmation and actual candidate/ranking build remain open.
> The unchanged recipe's February/March confirmation is now running: controller
> PID 73236, pipeline PID 78840. Its first tuned CatBoost is saved with native replay
> PASS, fitting all 1,417,542 permitted Lane A rows over 244 days and querying every
> ordered February/March Lane A ID, 305,989 rows. Actual cold bytes, feature/input
> fingerprints, source/configuration, native model and prediction bytes, stopping
> scopes, complete refit coverage and finite nonnegative outputs pass:
> `python analysis/champion_checkpoint_audit.py --months 2-3`.
> Proof: 2-3/first-member-audit.json. Budgets: 2,065/2,253/5,976; saved selection,
> refit and replay wall cost: 1,838.516 seconds. Its Lane A RMSE is 203.3926 seconds,
> a member diagnostic rather than a whole-file confirmation result or board gain.
> January/July and February/March labels are excluded from all fits. The first full
> original neighbour donor now has all twelve C/L/N components saved with checkpoint
> replay PASS across group-4-8-11, group-5-9-12, group-6-10 and the full fit.
> Actual cold bytes, permitted/query/stopping scopes, native model and prediction
> bytes, source/configuration, raw/floored blend arithmetic and saved neighbour
> feature reconstruction pass: `python analysis/champion_donor_audit.py --months 2-3`.
> Proof: 2-3/first-donor-audit.json. It fits 1,417,542 Lane A rows over 244 days; its
> feature file covers all 2,085,047 ordered cold rows. Output SHA:
> a06753dd99936740b4c71348369f21c2e762941ca5014f22b834429835ef0c21.
> Saved component wall cost is 4,103.734 seconds, with 3,579.093 GPU-family seconds,
> including selection/refit/replay. Matching original empty-window missing positions
> reproduce; infinities are rejected. Native replay is retained from completion records;
> the read-only audit ran no new fit or GPU replay. This donor stage is checked.
> The outer Lane A members are now checked; original mixture B's full audit, nested meta chains
> and full-block evaluation remain pending.
> The separate downstream stopping-day-excluded donor is also saved and checked:
> `python analysis/champion_donor_audit.py --months 2-3 --selection-donor`.
> Proof: 2-3/selection-donor-audit.json. All twelve C/L/N components have native replay
> PASS. Actual cold bytes, ordered scopes, training-only stopping, source/configuration,
> native model/prediction bytes, raw/floor arithmetic and neighbour feature reconstruction
> pass. Every donor fit excludes the downstream member's 132,380 stopping rows over
> 25 days as well as January/July and February/March labels. Full fit: 1,285,162 Lane A
> rows over 219 days; ordered feature rows: 2,085,047. Output SHA:
> 3621a1529ef74d7d1c166c1947598afe5f701186fc3af5b5ae223c2e097c2ea3.
> Saved component wall cost: 3,404.891 seconds; GPU-family proxy: 2,922.797 seconds.
> No new model fit or GPU replay ran in this read-only audit. Both outer donor chains
> are checked and priced. The tuned LightGBM is now saved and checked:
> `python analysis/champion_checkpoint_audit.py --months 2-3 --member ra_lgb_pw_t7`.
> Proof: 2-3/ra_lgb_pw_t7-audit.json. Actual cold bytes, ordered fit/query/stopping
> scopes, full row/day refit coverage, native model/prediction bytes, source/configuration,
> finite nonnegative outputs and retained native replay pass. Its fit, query and
> stopping-input fingerprints reproduce from cold features and both checked donors.
> It fits all 1,417,542 permitted Lane A rows over 244 days and predicts all 305,989
> ordered February/March Lane A IDs. Selected budgets: 219/6,335/7,999. Saved selection,
> refit and replay wall cost: 1,960.219 seconds on CPU. Output SHA:
> 7b728cd288d4a321d866b19d7c5d2191c1d94f52642acb384c4319c976886d55.
> Lane A RMSE is 208.7182 seconds, a member diagnostic rather than whole-file
> confirmation or a board gain. The scored champion and its frozen copy retain their
> recorded hashes. The tuned neural is also saved and checked:
> `python analysis/champion_checkpoint_audit.py --months 2-3 --member nn3_pw_t7`.
> Proof: 2-3/nn3_pw_t7-audit.json. Actual cold bytes, fit/query/stopping scopes,
> native model/prediction bytes, source/configuration, full row/day refit coverage,
> finite nonnegative outputs and retained native replay pass. Its fit, query and
> stopping-input fingerprints reproduce from cold features and both checked donors.
> Training-only selection stops at epoch 79, selecting epoch 64; the all-row refit
> fits 1,417,542 Lane A rows over 244 days for exactly 64 epochs and predicts all
> 305,989 ordered February/March Lane A IDs. Saved selection, refit and replay wall
> cost: 651.797 seconds. Output SHA:
> 8196d6cb27947d0c353e8aca3c70b0b0f7940737293a1ebcbd99c7e951db5e61.
> Its Lane A RMSE is 220.1323 seconds, worse than the standalone CatBoost diagnostic;
> this is an input to the unchanged blend, not a whole-file confirmation or board gain.
> All three outer Lane A members and both outer donor chains are saved and checked.
> The original mixture B is now saved with native replay PASS in its completion record,
> at 409.344 seconds and selected budgets 2,161/3,041. It fits 1,432,447 permitted
> all-lane rows over 244 days and queries 2,192 Lane B rows. Its full scope/native-byte
> audit closes with the whole-block recording/evaluation proof; that remains pending.
> The meta-4-8-11 tuned CatBoost is now saved with native replay PASS, fitting
> 893,511 permitted Lane A rows over 153 days and querying 524,031 inner Lane A rows.
> Saved selection/refit/replay wall cost: 1,421.313 seconds; budgets: 1,594/2,846/4,459.
> Its completion record is meta-4-8-11/ra_cb_wz_t7/complete.json. The existing assembly
> reader's native_record check passes actual source/configuration and all native model
> and prediction bytes for this newly saved member. Prediction SHA:
> f42d7f0f6e54ab75564d6f3d4ba5462daec435c4f18ba5c4a8fcf0959d8bd9f4.
> Its scope/raw-value checks now pass in the complete inner-table proof below.
> This inner scope's full original neighbour donor is now saved and checked:
> `python analysis/champion_donor_audit.py --months 2-3 --meta-group 4 8 11`.
> Proof: 2-3/meta-4-8-11-donor-audit.json. All twelve C/L/N components have native
> replay PASS across group-5-10, group-6-12, group-9 and the full fit. Actual cold bytes,
> fit/query/stopping scopes, full refit rows/days, native model/prediction bytes,
> source/configuration, raw/floor blend arithmetic and neighbour feature reconstruction
> pass. January/July, February/March and inner April/August/November labels are excluded
> from all donor fits. The full fit covers 893,511 Lane A rows over 153 days; its saved
> neighbour features cover every ordered cold row, 2,085,047. Output SHA:
> 0e4006be44e6aaa9baa50a78aaefbf24ef274ca2884ad434b1750dfb15801088.
> Saved selection/refit/replay wall cost: 3,262.110 seconds; GPU-family proxy:
> 2,493.391 seconds. Matching original empty-window missing positions reproduce;
> infinities are rejected. This read-only audit ran no new fit or GPU replay.
> The separate downstream stopping-day-excluded donor is now saved and checked:
> `python analysis/champion_donor_audit.py --months 2-3 --selection-donor --meta-group 4 8 11`.
> Proof: 2-3/meta-4-8-11-selection-donor-audit.json. All twelve C/L/N components retain
> native replay PASS across group-5-10, group-6-12, group-9 and full. Actual cold bytes,
> fit/query/stopping scopes, full refit rows/days, native model/prediction bytes,
> source/configuration, raw/floor arithmetic and neighbour feature reconstruction pass.
> Every fit excludes the downstream member's 83,735 stopping rows over 16 days, plus
> January/July, February/March and the inner April/August/November query labels.
> Full fit: 809,776 Lane A rows over 137 days; feature output covers all 2,085,047
> ordered cold rows. Output SHA:
> 9efc3c6579ba5f9d124d3e47185d8a05b3df8f0818a39add27328f520a99ea41.
> Saved selection/refit/replay wall cost: 2,928.891 seconds; GPU-family proxy:
> 2,268.578 seconds. Original empty-window missing positions reproduce; infinities fail.
> This read-only audit ran no new fit or GPU replay. Both first-inner donor chains
> are checked and priced. The tuned inner LightGBM is now saved with native replay
> PASS: 1,432.234 seconds and budgets 133/5,246/7,960. The tuned inner neural is also
> saved with native replay PASS: 179.735 seconds, selection stopped at epoch 45 and
> selected epoch 30, followed by exactly 30 all-row refit epochs. Both members fit
> 893,511 Lane A rows over 153 days and query 524,031 inner Lane A rows. The original
> inner mixture B is saved with native replay PASS: 370.765 seconds, budgets
> 1,333/3,947. Its fit covers 903,646 all-lane rows; query covers 4,770 Lane B rows.
> The first February/March inner raw table is now saved and checked:
> `python analysis/champion_assembly_audit.py --months 2-3 --meta-group 4 8 11`.
> Proof: 2-3/meta-4-8-11-assembly-audit.json. Actual complete April/August/November
> files cover 528,801 query rows: 524,031 Lane A and 4,770 Lane B. Ordered fit/query
> scopes, actual native model/prediction bytes, source/configuration, stopping coverage,
> raw/floor parity, member assembly and identical original mixture B in all three
> columns pass. Output SHA:
> cc140cf936eb227e312bbf079498daddcfc0532b2d54b9cced3ed0565d159838.
> This assembly check does not reproduce fit/query feature fingerprints or donor windows;
> the two donor window proofs are already checked separately. Full NNLS, residual
> recalibration, final recording-B override and whole-file errors remain pending.
> The meta-5-9-12 tuned CatBoost is now saved with native replay PASS: 1,104.047
> seconds and budgets 1,053/897/4,576. It fits 888,432 Lane A rows over 152 days and
> queries 529,110 inner Lane A rows. Completion record:
> 2-3/meta-5-9-12/ra_cb_wz_t7/complete.json. The assembly reader's native_record check
> passes actual fitting source/configuration and native model/prediction bytes.
> Prediction SHA:
> 07d50921c9965fa1848569df08648db2c8404ac385f07234de922520557e2eab.
> Its fit/query/stopping scopes and raw-value lineage close with this group's complete
> inner-table proof, still pending. No new fit or GPU replay ran in the native reader.
> This group's full original neighbour donor is now saved and checked:
> `python analysis/champion_donor_audit.py --months 2-3 --meta-group 5 9 12`.
> Proof: 2-3/meta-5-9-12-donor-audit.json. All twelve C/L/N components retain native
> replay PASS across group-4-10, group-6-11, group-8 and full. Actual cold bytes,
> fit/query/stopping scopes, full refit rows/days, native model/prediction bytes,
> source/configuration, raw/floor arithmetic and neighbour feature reconstruction pass.
> Every fit excludes January/July, February/March and inner May/September/December
> labels. Full fit: 888,432 Lane A rows over 152 days; saved features cover all
> 2,085,047 ordered cold rows. Output SHA:
> 056e22e0b86c2da7986607815e1f8e7e22aacf124360782a9f77e1efbb15043c.
> Saved selection/refit/replay wall cost: 2,621.876 seconds; GPU-family proxy:
> 1,970.641 seconds. Original empty-window missing positions reproduce; infinities fail.
> This read-only audit ran no new fit or GPU replay. The full second-inner donor stage
> is checked and priced. Its separate stopping-day-excluded donor is now saved and checked:
> `python analysis/champion_donor_audit.py --months 2-3 --selection-donor --meta-group 5 9 12`.
> Proof: 2-3/meta-5-9-12-selection-donor-audit.json. All twelve C/L/N components retain
> native replay PASS across group-4-10, group-6-11, group-8 and full. Actual cold bytes,
> fit/query/stopping scopes, full refit rows/days, native model/prediction bytes,
> source/configuration, raw/floor arithmetic and neighbour feature reconstruction pass.
> Every fit excludes the downstream member's 84,701 stopping rows over 16 days, plus
> January/July, February/March and inner May/September/December query labels.
> Full fit: 803,731 Lane A rows over 136 days; features cover all 2,085,047 ordered
> cold rows. Output SHA:
> 3ad75aa18e0a2e8cd271dadb05cd4ae6f181ec691df37ab734f32a53f11d193d.
> Saved selection/refit/replay wall cost: 3,099.721 seconds; GPU-family proxy:
> 2,791.501 seconds. Original empty-window missing positions reproduce; infinities fail.
> This read-only audit ran no new fit or GPU replay. Both second-inner donor chains
> are checked and priced. The tuned meta-5-9-12 LightGBM has reached full refit;
> its fit scope covers 888,432 Lane A rows over 152 days and its query covers 529,110
> inner Lane A rows. Actual process CPU time advances; error is null. Controller and
> pipeline command lines and actual logs were checked. The whole inner table is pending.
> Remaining inner tables and full-block evaluation remain pending. Existing proofs
> are preserved; no serial fit was restarted.
> Current controller/pipeline command
> lines were checked directly; the serial pipeline continues without restarting fits.
> Logs: pipeline-2-3.out/.err and controller-2-3.out/.err. Verify actual processes.
> The pre-fit declaration is appended to CLEAN-CHAMPION-2026-10-03.html. Block cap:
> 12 GPU-family hours in 2-3/budget-limit.json; reserve 16 hours for final preparation.
> September/October remains unrun and deferred: this is reduced confirmation coverage.
> The global 44-hour and block cap are enforced at saved-component boundaries; four
> budget tests pass. The controller accepts the declared block and preserves first-run
> logs. Its first launch check matched its own shell command; filtering Python processes
> corrected that check before any job started. No fitting recipe changed.
> Audit readers now accept `--months 2-3` and validate inner groups against that outer
> scope. All new confirmation proofs go under 2-3/, retaining the existing first-block
> paths and bytes. Donor and full-evaluation readers reject proofs from another block.
> Donor checks now verify actual native artifact bytes, fitting source and configuration.
> Eleven focused scope/native-byte tests and all four CLI entry points pass; current F
> lint passes. No fitting module or recipe changed. These reader checks are complete;
> actual later donor/inner/full evaluations remain pending until their artifacts finish.
> Confirmation inner groups are 4/8/11, 5/9/12 and 6/10. When a donor finishes, run
> `python analysis/champion_donor_audit.py --months 2-3` (add `--selection-donor` or
> `--meta-group` for its declared scope). When an inner table finishes, run
> `python analysis/champion_assembly_audit.py --months 2-3 --meta-group <group>`.
> Close the whole block with `python analysis/champion_evaluation_audit.py --months 2-3`.
> New component proofs export separately in the local status page. Budget.json now
> records 66,193.3375 GPU-family seconds globally, with 22,001.518 in the confirmation
> block; both declared limits remain in force. Reprice from actual completed records.
> The campaign remains OPEN. Inspect each completed block and continue the approved
> work; a completed component, failed candidate or expected budget stop does not end it.

> **Resumed score work — 2026-10-03.** Keith reiterated his instruction to keep improving
> the score after the session stopped too early on the failed MSE lead. The task remains open.
> The original tuned CatBoost January/July member completed with saved-model replay passing.
> Its restart PID 27196 exited successfully. Controller PID 59744 started the full serial
> pipeline, PID 52408, running `analysis/champion_screen.py --months 1-7`. The first original
> donor's three cross-fit scopes, group-2-5-9-12, group-3-6-10 and group-4-8-11, and the
> full-scope donor now have all three saved members each with checkpoint replay PASS.
> The separate stopping-day-excluded donor is now saved: all three cross-fit groups and
> its full-scope C/L/N members have checkpoint replay PASS. All three outer Lane A members
> are now saved with checkpoint replay PASS. The original mixture-B donor is also saved
> with replay PASS, at 841.0 seconds of component cost. The pipeline has written the
> floored outer complete-file member table. Meta-2-5-9-12's tuned CatBoost is now saved
> with checkpoint replay PASS, covering 1,051,799 permitted Lane A rows over 183 days
> at selected budgets 1,453/1,595/5,461 and 1,468.9 seconds of component cost. Its nested
> PW donor has saved all three members in group-3-8, group-4-10, group-6-11 and the
> full scope with replay PASS. Its actual cold-input bytes, ordered scopes, stopping
> coverage, saved prediction bytes and PW window reconstruction pass:
> `python analysis/champion_donor_audit.py --meta-group 2 5 9 12`.
> meta-2-5-9-12-donor-audit.json records 3,204.5 seconds of total component wall time
> and 2,734.1 seconds of GPU-family wall time. January/July and the inner query months
> are excluded from every fit. The reader also rejects undeclared inner query groups.
> The separate stopping-day-excluded nested donor now has all twelve members saved
> with replay PASS. Its actual scopes, cold inputs, stopping coverage, prediction bytes
> and window reconstruction pass:
> `python analysis/champion_donor_audit.py --selection-donor --meta-group 2 5 9 12`.
> meta-2-5-9-12-selection-donor-audit.json records 3,645.2 seconds of component wall
> time and 3,236.8 seconds of GPU-family wall time. It fits 950,561 rows over 164 days,
> excluding 101,238 downstream stopping rows over 19 days as well as both query scopes.
> Both first-meta donor chains are now checked and priced. All three first-meta tuned
> members and the original raw mixture-B donor are now saved with native replay PASS.
> The first inner table covers all 678,561 ordered rows in the complete February, May,
> September and December files, including 671,732 Lane A and 6,829 Lane B rows.
> The assembly audit checks actual cold bytes, permitted/stopping/query scopes, native
> model/prediction bytes, configurations, fitting source hashes, raw/floor parity,
> exact raw member-column values and identical independently fitted raw mixture B.
> Command: `python analysis/champion_assembly_audit.py --meta-group 2 5 9 12`.
> Record: meta-2-5-9-12-assembly-audit.json. The reader does not reproduce fit/query
> feature fingerprints or donor windows; the donor windows have separate proofs above.
> This closes one inner table, while NNLS/recalibration/final B/evaluation remain pending.
> Direct process, live job and log checks at 12:26 London on 4 October confirm the
> controller and pipeline continue without a recorded error. Meta-3-6-10's tuned
> CatBoost is saved with native replay and raw/floor parity PASS, covering 1,195,763
> permitted Lane A rows over 211 days at budgets 1,667/3,476/5,982 and 2,040.7 seconds
> of component cost. Its nested full donor now has all twelve C/L/N components saved
> with native replay PASS, covering group-2-8-12, group-4-9, group-5-11 and the full fit.
> Actual cold-input bytes, ordered permitted/query scopes, stopping coverage, saved
> prediction bytes, raw/floored blend arithmetic and PW window reconstruction pass:
> `python analysis/champion_donor_audit.py --meta-group 3 6 10`.
> meta-3-6-10-donor-audit.json records 4,107.8 seconds of total component wall time
> and 3,582.9 seconds of GPU-family wall time. It fits 1,195,763 Lane A rows over 211
> days, excluding January/July and the inner March/June/October query files from every
> donor fit. Empty-window missing positions reproduce; infinities are rejected.
> The separate stopping-day donor now has all twelve C/L/N components saved with
> native replay PASS. Actual cold-input bytes, ordered scopes, stopping coverage,
> saved prediction bytes, raw/floored blend arithmetic and PW window reconstruction
> pass: `python analysis/champion_donor_audit.py --selection-donor --meta-group 3 6 10`.
> meta-3-6-10-selection-donor-audit.json records 4,156.0 seconds of component wall
> time and 3,761.9 seconds of GPU-family wall time. It fits 1,079,484 rows over 189 days,
> excluding 116,279 downstream stopping rows over 22 days as well as January/July
> and the inner March/June/October query labels. Both second-meta donors are checked
> and priced. The tuned LightGBM is now saved with native replay PASS, covering
> 1,195,763 permitted Lane A rows over 211 days at budgets 189/6,788/7,923 and
> 2,335.8 CPU wall seconds. Its raw/floor parity now passes. The tuned neural is saved
> with native replay and raw/floor parity PASS: selected epoch 56, selection stopped
> at 71, all 1,195,763 permitted rows refitted at 56, at 646.8 GPU-family wall seconds.
> Original raw mixture B is also saved with native replay PASS, fitting all 1,207,362
> permitted rows and querying 5,498 Lane B rows, at 258.8 seconds of component cost.
> The second inner table covers all 533,266 ordered rows in the complete March, June
> and October files, including 527,768 Lane A and 5,498 Lane B rows. Actual cold bytes,
> permitted/stopping/query scopes, native model and prediction byte hashes,
> configurations, fitting source hashes, raw/floor parity, exact raw A columns and
> identical raw mixture-B columns pass:
> `python analysis/champion_assembly_audit.py --meta-group 3 6 10`.
> Record: meta-3-6-10-assembly-audit.json. This reader does not reproduce fit/query
> feature fingerprints or donor windows; both second-meta donors have separate window
> proofs above. Meta-4-8-11's tuned CatBoost is now saved with native replay and
> raw/floor parity PASS, fitting 1,199,500 permitted Lane A rows over 212 days at
> budgets 2,293/2,458/5,845 and 2,043.2 seconds of component cost. Records:
> 1-7/meta-4-8-11/ra_cb_wz_t7/complete.json and raw_predictions.json in the run.
> Its full nested neighbour donor now has all twelve C/L/N components saved with
> native replay PASS, covering group-2-6-12, group-3-9, group-5-10 and the full fit.
> Actual cold-input bytes, ordered permitted/query scopes, stopping coverage, saved
> prediction bytes, raw/floored blend arithmetic and PW window reconstruction pass:
> `python analysis/champion_donor_audit.py --meta-group 4 8 11`.
> meta-4-8-11-donor-audit.json records 4,416.0 seconds of total component wall time
> and 3,944.5 seconds of GPU-family wall time. It fits 1,199,500 Lane A rows over
> 212 days, excluding January/July and the inner April/August/November query labels
> from every donor fit. Matching original missing windows reproduce; infinities fail.
> The separate stopping-day donor now has all twelve C/L/N components saved with
> native replay PASS across the same three groups and its full fit. Actual cold-input
> bytes, ordered permitted/query scopes, stopping coverage, saved prediction bytes,
> raw/floored blend arithmetic and PW window reconstruction pass:
> `python analysis/champion_donor_audit.py --selection-donor --meta-group 4 8 11`.
> meta-4-8-11-selection-donor-audit.json records 3,915.5 seconds of component wall
> time and 3,474.8 seconds of GPU-family wall time. It fits 1,083,221 rows over 190
> days, excluding 116,279 downstream stopping rows over 22 days as well as January/July
> and the inner April/August/November query labels. Both third-meta donors are checked
> and priced. Its tuned LightGBM is now refitting all 1,199,500 permitted rows over
> 212 days, with no recorded error. Its tuned neural and raw mixture-B members,
> third complete-file assembly, whole-file NNLS, residual recalibration, final
> recording B and outer evaluation remain pending.
> Two inner tables are checked.
> The current controller-resume-1.err
> is empty; controller.err retains the earlier failed controller attempt's traceback.
> Budget.json records 43,116.3 seconds of completed
> GPU-family component cost, within the declared allowance.
> LightGBM and neural completion records cover every 1,723,531 permitted Lane A row over
> 303 days. LightGBM selected budgets 217/6,312/7,991; the neural selected epoch 50, stopped
> selection at 65 and refitted all rows at 50. Both new member audits pass actual model and
> prediction byte hashes, configurations, fitting source hashes, cold input bytes,
> ordered fit/query IDs, stopping coverage, complete refit coverage and finite nonnegative
> output coverage. Their fit, query and selection feature fingerprints reproduce from the
> cold features and the two checked outer PW files. Commands:
> `python analysis/champion_checkpoint_audit.py --member ra_lgb_pw_t7` and
> `python analysis/champion_checkpoint_audit.py --member nn3_pw_t7`.
> Records: ra_lgb_pw_t7-audit.json and nn3_pw_t7-audit.json under the clean-champion run.
> Their saved selection/refit/replay costs are 2,759.3 CPU wall seconds and 820.3 GPU-family
> wall seconds respectively. The scored champion and its frozen copy still match their
> recorded hashes. Native replay evidence is retained from completion records; these
> audits perform no new model fit or GPU replay. Member errors are descriptive Lane A
> diagnostics, not a whole-file ensemble comparison or scorer gain.
> `python analysis/champion_donor_audit.py` passes actual cold-input bytes, permitted/query
> IDs, stopping coverage, saved prediction hashes and reconstruction of the saved PW
> features from raw cross-fit/floored full predictions, including matching empty-window
> missing positions and rejecting infinities. An initially overstrict finite-feature check
> rejected the original per() empty-window NaNs; the audit now follows that declared rule.
> Training was unaffected. Its first-donor-audit.json records
> 6,182.4 seconds of total component wall time and 5,399.8 seconds of GPU-family wall time.
> The separate downstream stopping-day donor is checked with
> `python analysis/champion_donor_audit.py --selection-donor`. Actual fit/query hashes,
> cold inputs, stopping coverage, prediction bytes and window reproduction pass. Its fit
> excludes 164,028 downstream stopping rows across 31 days as well as January/July labels.
> selection-donor-audit.json records 5,976.0 seconds of total component wall time and
> 5,266.7 seconds of GPU-family wall time. Native replay PASS is retained from the saved
> completion records; this read-only audit does not repeat a model fit or GPU replay.
> Both GPU-family subtotals are conservative compute proxies, not measured GPU utilization.
> All nested meta donor costs are now checked; the full reference comparison is pending. This is component
> progress, not a full reference result or score gain. Verify the current active record
> and actual processes.
> It writes `data/runs/clean-champion-20261003/active.json` and the live HTML job record.
> Codex heartbeat `continue-prc26-score-campaign` is ACTIVE, attached to this chat,
> checking every half hour. It should repair failed fits, inspect completed references,
> continue confirmation or the next declared experiment, and stay quiet on unchanged work.
> The first heartbeat wakeup was observed in this chat and performed the completion/handoff
> check. Later wakeups still need actual process/log verification.
> Recheck these processes and records before treating this dated checkpoint as current.
> Plan: [CLEAN-CHAMPION-2026-10-03.html](CLEAN-CHAMPION-2026-10-03.html).
> The reviewed rebuild restores the original full C/L/N neighbour donor, raw OOF arithmetic,
> original mixture-B inputs for whole-file NNLS, six-decimal applied weights, fixed residual
> recalibration and a separately scoped final recording-B override. Stopping-day donors
> are rebuilt separately. Lane B day/window context is now file-aware. Cache fingerprints
> include permitted targets, inputs, source/weather provenance, code and package versions;
> model replay precedes component completion. Raw predictions actually used are saved.
> The first attempt saved its full classifier and then failed when CatBoost returned None
> for best iteration without validation. Both wrappers now return the fixed budget in that
> case. Atomic native saves and checked partial restart are added. Recovery verifies the
> original source bytes, unchanged inputs/scopes/configuration/dependencies, selected tree
> counts/features and saved-model hashes. Selected budgets and the full classifier are
> reused; the unfinished tail/body regressors continue. Prior fit time is charged to the
> compute record. Logs: outer-C-resume-1.out/.err; controller-resume-1.out/.err. First attempt
> logs and original selection provenance are retained. Native restart, scope/hash rejection
> and CatBoost without validation have regression checks. Both slash formats are covered;
> the path-check correction's explicit compatibility entry retains the actual running fit's
> engine hashes while allowing that verified component to enter the next queued stage.
> It does not relax input, label, configuration, dependency or artifact checks. All three
> refit heads are now saved; complete.json records checkpoint replay PASS. The queued
> pipeline successfully reused this component after its fingerprint/artifact checks.
> `python analysis/champion_checkpoint_audit.py` checks actual cold-input bytes, allowed
> fit/stopping/query IDs, all-row/day coverage, prediction IDs/finiteness/floor and the
> preserved champion. first-member-audit.json records PASS, Lane A-only descriptive errors
> and 2,704.4 seconds of saved-component cost including the interrupted first attempt.
> January/July Lane A member RMSE is 222.2729 s, versus AOBT-only 410.2841 s; this is a
> reused member diagnostic, not an assembled whole-file ensemble or a new scorer result.
> Both outer donor chains and all nested meta donors are checked and priced; the full
> comparison, weather/episode checks and confirmation remain open.
> Full suite: 73 passed; current F lint passes. The baseline and its comparison remain
> incomplete while the fits run. The controller currently covers January/July only, then
> the main session must inspect results, price the chain, continue confirmation or declare
> the next distinct experiment. No upload or new scorer gain. Do not end the campaign at
> a component or candidate rejection. The reference checks a 44-hour GPU-family allowance
> inside the existing campaign envelope, at saved-component boundaries.

> **Continuation checkpoint — 2026-10-03.** Eight initial recipes, including
> the isolated CPU control, are declared in CONTENDER-RUN-2026-10-03.md.
> CP1's three mechanisms fail the winter screen; normalized weather has a
> small uncertain gain; signed recorder experts harm winter and hot rows.
> Corrected sampled MSE beats Huber in both development blocks. A nested
> sampled blend gives autumn gains but weak winter results, below the pooled
> promotion floor and with winter-weather harm. No scorer upload or promotion.
> `python analysis/neural_screen.py full` was stopped after its first full block: all permitted
> outer fit days, original CELL_T7 cap/patience/optimizer, the same nested
> 80-tree proxy donor and both declared losses. MSE loses to Huber by 1.8848 s,
> day interval [-3.0016, -0.8151], five-day [-2.8262, -1.0511], reversing both
> winter months. Declared reversal rule rejects the lead; second full block
> and conditional full-MSE meta check are not completed. No training job remains
> from this run. Preserved files: data/runs/contender-20261003/neural-full/.
> This comparison is not the full clean-v53 reconstruction. Full confirmation
> folds, champion donor/meta lineage, upload checks and release rehearsal remain.
> 62 tests pass in the full suite; two focused meta checks and 26
> optimized-Python submission tests pass. Static F lint passes. Real Lane B excluded-label mutation,
> missing-query-target, stopping-call and saved-control reproduction pass.
> Cold features are per supplied file; two UTC month-boundary cases are scoped
> by file membership. No claim of faulty clocks is supported.
> Current board snapshot: 2026-10-03 16:54 UTC, v53 270.43 s, rank 51/222;
> leader strong-crane 216.5675 s. Champion and global prediction caches preserved.
> Updated local report: [STATUS-2026-10-03.html](STATUS-2026-10-03.html).
> Submission guard equivalents from unmerged PR #1 are applied locally;
> the actual preserved champion passes normal and optimized Python checks.
> Full Huber's first member refits all 1,417,542 rows across 244 days at epoch
> 67, with checkpoint encoder/target scale and donor ID hashes matching its
> declared scope. Its first whole-file gain over the CPU control is -2.0572 s.
> Full MSE refits the same rows/days at epoch 69, selection stops at 84.
> Both checkpoint prediction replays pass at 1e-4 s absolute tolerance plus
> 1e-7 relative tolerance; actual IDs and unchanged Lane B pass. Both networks
> lose on winter weather despite improving ordinary rows. This comparison is
> one full block, not complete CV or a champion comparison.
> Ranking-support audit: five-day-grouped matched year AUC .8073; January
> .6466, July .9222. Severe snow-report counts are absent from matched 2025
> Jan/July but present in other training months and ranking. No density weights.
> Publication preflight: task Ready at 21:30 UTC on October 11; script/action,
> authenticated read access and campaign2 existence checked. Task requires
> interactive login and AC. Future execution is not verified; no visibility
> command was run. Private repo/default master remain live. Local changes
> still need to reach the intended release branch. Record: release_preflight.json.
> Local correctness commits: 1f2f3cb (fit scopes/all-row refits) and 68645a0
> (explicit submission guards). Research and results are saved with this checkpoint. Legacy global
> PW/tune/recal/blend paths remain outside the scoped campaign proof; complete
> clean-v53 is still unfinished. Do not reopen rejected recipes without a new
> prospective declaration and a distinct source of signal.

> **Current execution update — 2026-10-03.** Local branch `codex/validation-scope-20261003`,
> based on `9d0bc36`; changes are local and uncommitted. Live board: v53 remains best at
> 270.43 s, rank 51/222; strong-crane leads at 216.5675 s. This machine has the data and
> a working RTX 5080, with torch 2.10.0+cu128 (not the requirements.txt torch pin).
> Lane B history now requires permitted training IDs; each CV fold excludes its entire
> outer month block across both lanes. Stopping budgets come from an inner training split,
> followed by an outer-training refit. Tests: `python -m pytest tests/ -q`, 22 passed.
> `python analysis/laneb_scope_diagnostic.py` regenerates the diagnostic and preserved v53
> manifest under `data/runs/cp0-laneb-scope-20261003/`, without changing global caches.
> Fixed-budget history correction: full-file reused holdout 304.8745 to 304.6092 s,
> gain 0.2653 [0.0258, 0.5974]. Historical Lane A/donor caches remain fixed, so this is
> not the fully clean baseline or a new board gain. Real-data hidden-label mutation passes.
> Next: finish Lane A stopping, neighbour-prediction and meta/recalibration scopes,
> cold ranking-like masking/boundaries, then clean-v53. CP1–CP4 have not started.
> Exact champion bytes and Lane B parity pass; fitted-model lineage remains incomplete.
> Publication task is Ready for 11 October at 22:30 London; no rehearsal or upload here.
> Full status: [STATUS-2026-10-03.html](STATUS-2026-10-03.html). Submission sign-off still applies.


> **Current execution update — 2026-10-02.** Follow [CONTENDER-PLAN.md](CONTENDER-PLAN.md),
> CP0–CP5, and protocol amendment v9. Start with fit-scope isolation and a clean baseline:
> the review confirmed held-out-label dependency in Lane B copy history, scored-fold stopping
> and incomplete nested feature/meta isolation. The NN also needs a true all-data refit.
> v53 (270.43 s) remains the latest recorded champion; preserve the exact file and manifest.
> Historical sections below include completed jobs, stale “next” steps and stronger validation
> claims than the code supports. They are history, not a current job-status report.
> CP1 targets calibrated catastrophic recording mechanisms; CP2 conditional Lane A disruption;
> CP3 the full-data NN; CP4 the nested ensemble; CP5 final build/release.
> No new training or scorer upload was executed by the plan commit. Per-upload sign-off,
> the standing instruction to wait for unrelated GPU jobs, and the approved publication task remain.

Written mid-campaign so the next session can pick up without re-deriving anything. Every number below has the command
that regenerates it. Nineteen days to the deadline (11 Oct 2026 23:59:59 CET, `PLAN.md:3`).

> **Update 2026-09-23.** Sections 1, 3 and 4 are done and out of date. v34 (tuned LightGBM and MLP, shipped CatBoost)
> is the file of record: board 270.5655, rank 28 of 178, NEW BEST by 0.99 s over v32 (`LEDGER.md`, v34 row). v36 (equal
> weights) and v37 (weights fit on the 2025 holdout) both lost to it, 270.5691 and 270.6994, so the member weights are
> not a lever (`LEDGER.md`, v36 and v37 rows). Each member alone lost too: v38 (LightGBM) 274.6858 and v39 (CatBoost)
> 273.4283, so the mix is worth 2.9 to 4.1 s. v40 (the MLP alone) uploads at 00:05 UTC on 2026-09-24 via Windows task
> `prc26-v40-20260924`, and v41 (v34 plus the B1 bet) at 00:15 UTC via `prc26-v41-20260924`; each score goes in a new
> `LEDGER.md` row. The top-3 gap review is
> https://claude.ai/artifact/GxGV3b1LuLZ8ZwP56SXYrC (`analysis/gap.py`, `analysis/board_history.py`).
>
> **Update 2026-09-24.** v35 and v41 were both scored and both lost to v34 (`LEDGER.md`, the v41 upload and v35 upload
> rows). v41 (v34 plus B1) scored 271.7314, 1.166 s worse, so **B1 is dead: drop it from every later file**, which
> undoes the "B1 rides every later file" declaration in the v41 row. v35 (PW root fix plus B1) scored 271.716. Next to
> v41 the fix is worth +0.015 s, so the PW fix is flat on the board. v40 was never uploaded. H1 is closed with no
> winner. v34 is still the file of record. Only one lever is left untried: `ra_cb_wz_t7`, the tuned CatBoost re-fit
> (`lanea.py cv ra_cb_wz_t7 mix_cb_wx`, then fit). It needs the GPU. At 17:30 UTC on 2026-09-24, fifty's
> `eval.score_tso_backtest` held it at 100%. On Keith's go it started anyway, beside fifty (pid 57436, `gpu_ram_part`
> 0.3 in `c3287b0`, logs `data/r7_cv_cb.out` / `.log`). Fold 0 scored 203.33 in 5,813 s, the best fold 0 of any member
> (tuned LightGBM 206.88, MLP 211.25). The whole cv run is about 10 h, so the 18.6-day estimate in the v34 row is wrong.
> When cv lands: `python lanea.py fit ra_cb_wz_t7 mix_cb_wx <its 3 iters>`, then reblend and gate as for v35. The body
> regressor stopped at 5,986 of the 6,000-iteration cap, so it may still be undertrained. The expected gain is
> small: the tuning gained +1.142 on the judge folds, and recent holdout gains carried to the board at 0.15 or less.
>
> **Update 2026-09-26.** `ra_cb_wz_t7` landed: holdout lane A 221.93 vs 223.71 shipped (`docs/RESULTS.md`). On Keith's
> instruction every 1-, 2- and 3-member mix of the tuned C, L, N (NNLS and equal weights) was built by `grid.py` and
> uploaded as v42-v51. **All 10 are void**: `recal.fit` took lane B from the old donor (`LEDGER.md`, v42-v51 row);
> fixed in `82e5161`, all 11 `data/grid/*.parquet` rebuilt, lane B now matches v34 on 0 of 5,290 differing rows. The bad
> files share one lane B, so their board order is a true lane A order, and one fixed upload gives every tag's true
> score as sqrt(board^2 - K). Queued: all 11 fixed files, 5 per UTC day from 00:01 UTC 2026-09-27, CLN and N first
> (`upload_queue.py` in the session scratchpad, pid 56024, log `data/grid/uploads.txt`). Then derive K from CLN, write every tag's true score to the ledger.
>
> **Update 2026-09-28.** v53 (CLN) is the file of record, 270.43, rank 38 of 196; v54-v62 all scored as predicted
> (`LEDGER.md`). L uploads as v63 at 00:01 UTC 2026-09-29, predicted 274.6906. `README.md` now rebuilds v53 and
> `requirements.txt` pins torch (`153a4a0`). Two checks for a lever we lack found none. `analysis/board_history.py`:
> each top-15 team made one upload worth 0.9 to 3.8B SSE; ours peaked at 0.18B. kind-mango's public repo (last commit
> 2026-09-26 11:32 UTC, `docs/AUDIT_PROMPT.md`) prices reading `AOBT_3_flt` at 254 to 271 s, and their next upload
> scored 270.58, so their 11 s jump was most likely switching on the field v34 has always used. The one lever that size
> is still the 43 LIRF hull rows (`docs/LANEB.md` section 12), which only board reads can settle; the ban stands.
> Ship stage: the GitHub repo is private and its default branch is the 2026-09-04 scaffold, so going public also means
> making `campaign2` the default. History holds no secrets (`git log --all -p`, `.env` never committed).
>
> **Going public, approved by Keith 2026-09-28.** Windows task `prc26-public-20261011` runs `data/go-public.cmd` at
> 21:30 UTC on 11 Oct (22:30 BST): default branch to `campaign2`, then public, logged to `data/go-public.log`. Any
> 11 Oct upload must land before then. The task wakes the PC and catches up if it was off. `gh` was proven to run from
> Task Scheduler with a read-only test. Afterwards check `gh repo view kitfunso/prc26 --json visibility`. A blob scan
> of all three remote branches (`campaign2`, the scaffold `master`, `v1`) found no `.env` value, key pattern or data
> file. The scaffold's `docs/data-viewer.html` holds 24 sample 2025 training rows. The site sets no data licence and no
> date for the code to go public.

## 1. Where we stand

| thing | value | source |
|---|---|---|
| file of record | `data/merry-octopus_v32.parquet` | `README.md:98` |
| board score | 271.5521, rank 20 of 152 | `LEDGER.md` v32 row |
| leader | vigorous-whistle 241.75 | same row |
| 10th place | strong-turtle 261.7928 | same row |
| v32 blend | `ra_cb_wz` 0.491 + `ra_lgb_pw` 0.175 + `nn3_pw` 0.334 | same row |
| v32 holdout | all 321.61, lane A 218.69, lane B 1900.6 | `grep "blend:ra_cb_wz+ra_lgb_pw+nn3_pw " docs/RESULTS.md` |
| scorer cap | 5 files a day, reset 00:00 UTC | `LEDGER.md:3` |

Upload rule, unchanged (`LEDGER.md:5`): expected board gain over 3 s against the current best, holdout CI excluding
zero, and `python board.py check` passes. Probing the scorer is banned.

## 2. What R7 was

Keith's instruction, verbatim: "for each of these three model, i need you to run proper hyperparameters tuning for all
parameters. show all parameters to me and the range. run optuna optimisation with 200 trials for each of them."

Every tunable parameter of all three v32 members, 200 Optuna trials each, judged on three folds the search never saw.
Search folds (2,3) and (4,5); judge folds (6,8), (9,10), (11,12). Studies live in `data/optuna.db` under
`<member>_full_23_45` and resume if killed. Full spaces and rationale: `docs/EXPERIMENT-PROTOCOL.md` section 4.

### Results: all three pass, 0 failed trials of 600

| member | COMPLETE / PRUNED / FAIL | judge folds: shipped to tuned | gain | transfer |
|---|---|---|---|---|
| `ra_lgb_pw` | 128 / 72 / 0 | 211.250 to 207.015 | **+4.235** | 118% |
| `ra_cb_wz` | 113 / 87 / 0 | 207.375 to 206.233 | **+1.142** | 81% |
| `nn3_pw` | 115 / 85 / 0 | 212.053 to 208.572 | **+3.481** | 74% |

Transfer is the judge gain over the tune gain. Above 100% means the winning region was not fitted fold noise.

Regenerate: `grep -c "Trial [0-9]* \(finished\|pruned\)" data/r7_*.log` and the `JUDGE` line in each `data/r7_*.out`.

### The re-fit that has landed

| member | holdout all | holdout lane A | lane A delta |
|---|---|---|---|
| `ra_lgb_pw` (shipped) | 326.37 | 225.75 | baseline |
| `ra_lgb_pw_t7` (tuned) | 323.67 | 221.75 | **-4.00** |

`grep -E "^\| ra_lgb_pw" docs/RESULTS.md`. The holdout lane A gain (4.00) lands within 0.24 of the judge-fold gain
(4.235), on rows neither the search nor the judge folds touched. Best transfer evidence in the campaign.

## 3. What is running right now

| job | pid | writes | check |
|---|---|---|---|
| `lanea.py cv ra_cb_wz_t7 mix_cb_wx` | 24108 | `data/r7_cv_cb.out` / `.log` | `tasklist //FI "PID eq 24108" //NH` |
| `nn.py cv nn3_pw_t7 mix_cb_wx`, queued behind it | launcher 53876 | `data/r7_cv_nn.out` / `.log` | starts when 24108 exits |

Both are detached processes, not harness background tasks, because the harness watchdog kills background tasks on a
low-memory alarm even when the machine has 28 GB free. Restart either with `Start-Process` and
`-RedirectStandardOutput` / `-RedirectStandardError`; never `2>&1`, a hook blocks it.

Expect the CatBoost re-fit to take several hours: the tuned settings drop the learning rate from 0.1 to 0.0417, so it
fits many more iterations. The LightGBM re-fit went from 641 s shipped to 7597 s tuned for the same reason.

## 4. Next steps, in order

1. Wait for both re-fits. Each appends its row to `docs/RESULTS.md`.
2. Reblend all three tuned members and compare against v32 in one command:
   `python blend.py ra_cb_wz_t7 ra_lgb_pw_t7 nn3_pw_t7 --base=blend_ra_cb_wz+ra_lgb_pw+nn3_pw`
   That runs NNLS on the OOF, scores the holdout, prints the paired day-block bootstrap delta and the regime slice.
3. Check each tuned member earns a non-zero NNLS weight. A member with weight 0.000 is dead regardless of its own
   holdout, which is exactly how `ra_ridge` died (`docs/EXPERIMENT-PROTOCOL.md:100`).
4. Refit the second stage (`recal.py`) on the new blend.
5. Apply the slot gate: section 3's 1.5 s all-rows bar, CI excluding zero, plus the `LEDGER.md:5` upload rule.
6. Present the Stage 6 surface including the Stage 4.5 coverage interrogation table, then **wait for Keith's sign-off.**
   Promotion requires explicit sign-off, every time. No scorer slot without it.

## 5. The honest read, before anyone gets excited

The judge-fold gains are real and clean. The board is a different question, and recent history is bad:

| file | holdout promise | board outcome |
|---|---|---|
| v29 | +0.873 [+0.104, +1.708] | **LOST** 0.43 s |
| v30 | +0.220 [+0.038, +0.423] | **LOST** 0.15 s |
| v31 | +0.240 [+0.108, +0.402] | won, carried 0.4 of promise |
| v32 | +0.754 [+0.392, +1.314] | won, carried 0.15 of promise |
| v33 | +0.402 [+0.109, +0.748] | **LOST** 0.29 s |

Three of the last five gated changes lost board seconds while passing a CI that excluded zero. The carry fraction on
the two that won fell from 0.4 to 0.15. This is the standing caution in `LEDGER.md:18-22` and it applies to R7 in full.

What that means concretely: a lane A holdout gain of 4.00 on a member carrying weight 0.175 in the blend is not 4 s of
board, and on the recent carry it may be a fraction of a second. The upload bar is 3 s. R7 on its own probably does not
clear it. The realistic read is that R7 is a foundation for the remaining lanes, not a submission by itself.

Also worth saying plainly: CatBoost bought +1.142 from the same 200 trials LightGBM turned into +4.235. Further
CatBoost tuning is priced as close to dead. Do not spend another search there.

## 6. Traps already paid for

- **Git Bash eats `/FLAG`.** `tasklist /FI "PID eq N"` reaches the exe as a mangled Windows path and fails. Use `//FI`.
  The worse half: a check built on a failed command reads as a real negative. Two watchers reported both jobs dead while
  both were alive. Test any liveness check against a case that must be negative before trusting it in a loop.
- **The harness background-task watchdog is not the machine.** It killed two jobs claiming low memory with 28 GB free.
  Long jobs go to `Start-Process`, detached.
- **Never hand-copy tuned parameters.** Both config sets were read back out of `data/optuna.db` through the same
  `space(fam, FixedTrial(best_params), base)` call `tune.py:148` uses for its judge run, then asserted equal to the
  config. Scripts are in the session scratchpad; the pattern is worth rebuilding if they are gone.
- **`tune.py` counts NEW trials per process.** Two workers on one study means 400 trials, not 200, which breaks the
  declared budget and inflates selection. Do not parallelise a search that way.
- **The GPU is shared** with another session's `fifty` jobs. Keith's standing choice is to WAIT, never pause them.

## 7. Commits

| commit | what |
|---|---|
| `48c6b36` | R7 search code for all three members |
| `1aa3e76` | tuned LightGBM and CatBoost configs, both tree results |
| `ecde569` | tuned MLP config, MLP result, first re-fit result |

All pushed to `origin master:campaign2`. Tests green: `python -m pytest tests/ -q`, 11 passed.
