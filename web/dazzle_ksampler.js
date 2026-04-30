/**
 * DazzleKSampler -- frontend extension shell.
 *
 * v0.1.3-alpha: the conditional seed-widget hiding logic that lived here in
 * v0.1.2-alpha has been removed. Two reasons:
 *
 *   1. The hide rule was based on a wrong premise. The original comment said
 *      "samples used as noise (Shape B); seed unused" -- but the seed is NOT
 *      unused. It still drives the per-step ancestral / SDE / guide-inversion
 *      noise sampler via `init_noise_samplers` in
 *      `py/beta/rk_noise_sampler_beta.py:135-160`. Hiding the seed widget
 *      misled users into believing the value was irrelevant when it actually
 *      drove visible per-step variation.
 *
 *   2. The hide mechanism (mutating widget.type / widget.computeSize)
 *      conflicted with ComfyUI's native widget->input conversion. When a wire
 *      was connected to the seed input AND any other widget value was
 *      changed, the seed widget vanished permanently. Recovery required
 *      deleting and recreating the node. Filed as #12.
 *
 * The fix for both issues converges on the same change: stop hiding the
 * widget. The user-facing communication of "seed still matters" now lives in
 * the structured first-step console banner (samplers.py dispatch site), the
 * latent_role tooltip text, and the README / wiki / CHANGELOG.
 *
 * For full context see:
 *   private/claude/2026-04-29__13-04-30__seed-still-affects-output-under-latent-role-noise.md
 *
 * The extension shell is preserved (registration still fires) so future
 * frontend-only enhancements have a stable import path. If/when we add a
 * `lock_to_upstream` widget (#TBD) that genuinely makes the seed unused, we
 * may revisit hide logic -- but with a safer mechanism that doesn't mutate
 * widget state behind ComfyUI's back.
 */

import { app } from "../../scripts/app.js";


app.registerExtension({
    name: "DazzleKSampler.FrontendShell",

    // Intentionally empty: no nodeCreated logic in v0.1.3-alpha.
    // Future widget orchestration can hook in here.
});
