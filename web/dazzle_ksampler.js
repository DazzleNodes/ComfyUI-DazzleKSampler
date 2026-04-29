/**
 * DazzleKSampler -- conditional seed-widget visibility.
 *
 * When latent_role indicates the seed will not be used (Shape B / C dispatch
 * paths consume an upstream noise tensor instead), collapse the seed widget
 * so the user is not misled into thinking it controls the result.
 *
 * Roles that hide the seed widget:
 *     "noise"              -- samples used as noise (Shape B); seed unused.
 *     "noise+latent_image" -- "noise" key used as noise (Shape C); seed unused.
 *
 * Roles that show the seed widget (default visible):
 *     "auto"               -- may dispatch any shape; show seed because
 *                             auto+SHAPE_A still uses the seed.
 *     "latent_image"       -- seed drives noise generation.
 *     "seed_driven"        -- seed drives noise generation (force-override).
 *
 * Dispatch-aware nodes that own a "seed" widget (this extension targets):
 *     DazzleKSampler, DazzleSharkSampler, DazzleBongSampler, DazzleTauSampler
 *
 * Nodes intentionally NOT targeted:
 *     DazzleKSampler_Chain    -- no seed widget (threaded internally).
 *     DazzleKSampler_Advanced -- advisory node, returns SAMPLER; latent_role
 *                                here is informational only and the seed is
 *                                still used by the downstream consumer.
 *     DazzleClownSampler      -- advisory node, same reasoning.
 */

import { app } from "../../scripts/app.js";

const DISPATCH_AWARE_NODES = new Set([
    "DazzleKSampler",
    "DazzleSharkSampler",
    "DazzleBongSampler",
    "DazzleTauSampler",
]);

const SEED_UNUSED_ROLES = new Set(["noise", "noise+latent_image"]);


function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (widget.origType === undefined) {
            widget.origType = widget.type;
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    } else {
        if (widget.origType !== undefined) {
            widget.type = widget.origType;
        }
        widget.computeSize = null;
    }
}


function syncSeedVisibility(node, roleWidget, seedWidget) {
    const role = roleWidget?.value ?? "auto";
    const hide = SEED_UNUSED_ROLES.has(role);
    setWidgetHidden(seedWidget, hide);
    // Preserve user-resized width; auto-fit height only. Calling
    // setSize(computeSize()) would clamp width back to the minimum,
    // collapsing nodes the user had widened (same bug Smart-Res-Calc hit).
    if (node.computeSize) {
        const computed = node.computeSize();
        const currentWidth = (node.size && node.size[0]) || computed[0];
        node.setSize([currentWidth, computed[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}


app.registerExtension({
    name: "DazzleKSampler.ConditionalSeedWidget",

    nodeCreated(node) {
        if (!DISPATCH_AWARE_NODES.has(node.comfyClass)) return;

        const roleWidget = node.widgets?.find(w => w.name === "latent_role");
        const seedWidget = node.widgets?.find(w => w.name === "seed");
        if (!roleWidget || !seedWidget) return;

        // Chain into any existing callback so we don't clobber other extensions.
        const prevCallback = roleWidget.callback;
        roleWidget.callback = function (value) {
            const result = prevCallback ? prevCallback.apply(this, arguments) : undefined;
            syncSeedVisibility(node, roleWidget, seedWidget);
            return result;
        };

        // Apply on workflow load / node creation so saved-state nodes start correct.
        syncSeedVisibility(node, roleWidget, seedWidget);
    },
});
