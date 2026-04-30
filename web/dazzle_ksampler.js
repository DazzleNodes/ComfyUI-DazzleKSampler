/**
 * DazzleKSampler -- frontend extension.
 *
 * v0.1.4-alpha adds the `noise_all` master selector. When set to anything
 * other than "custom", the three granular sub-widgets (noise_type_init,
 * noise_type_sde, noise_type_sde_substep) are hidden from the UI and their
 * values are forced to match `noise_all` server-side (see samplers.py
 * DazzleKSampler.main). When set to "custom", the sub-widgets reappear with
 * their previously-edited values.
 *
 * Implementation pattern: physical splice insert/remove on the node's
 * `widgets` array, with `widget.origType` preservation. Same pattern
 * SmartResCalc uses for fill_color visibility on output_image_mode -- proven
 * working across multiple widgets in that codebase. This avoids the
 * widget.type = "converted-widget" trick that broke #12 in v0.1.2.
 *
 * v0.1.3-alpha context (still applies):
 *   The conditional seed-widget hide rule from v0.1.2 is gone. The seed is
 *   NOT unused under latent_role={noise, noise+latent_image} -- it still
 *   drives the per-step ancestral / SDE noise sampler. See
 *   private/claude/2026-04-29__13-04-30__seed-still-affects-output-under-latent-role-noise.md
 */

import { app } from "../../scripts/app.js";


const SUB_WIDGET_NAMES = ["noise_type_init", "noise_type_sde", "noise_type_sde_substep"];


function findWidget(node, name) {
    return node.widgets ? node.widgets.find(w => w.name === name) : null;
}


function updateNoiseAllVisibility(node) {
    const noiseAllWidget = findWidget(node, "noise_all");
    if (!noiseAllWidget) return;

    const isCustom = noiseAllWidget.value === "custom";

    if (isCustom) {
        // Show the sub-widgets. Insert them right after noise_all in original
        // order, skipping any that are already in the widgets array.
        const noiseAllIndex = node.widgets.indexOf(noiseAllWidget);
        if (noiseAllIndex === -1) return;

        let insertAt = noiseAllIndex + 1;
        for (const name of SUB_WIDGET_NAMES) {
            const widget = node._dazzleSubWidgets ? node._dazzleSubWidgets[name] : null;
            if (!widget) continue;

            const existingIndex = node.widgets.indexOf(widget);
            if (existingIndex === -1) {
                // Restore saved value if we have one
                if (node._dazzleSubWidgetValues && node._dazzleSubWidgetValues[name] !== undefined) {
                    widget.value = node._dazzleSubWidgetValues[name];
                }
                node.widgets.splice(insertAt, 0, widget);
                widget.type = widget.origType || "combo";
                insertAt++;
            } else {
                // Already visible -- advance insert point past it
                if (existingIndex >= insertAt) {
                    insertAt = existingIndex + 1;
                }
            }
        }
    } else {
        // Hide the sub-widgets. Save their current values first, then remove
        // in reverse index order to avoid index shifts.
        if (!node._dazzleSubWidgetValues) node._dazzleSubWidgetValues = {};

        const toRemove = [];
        for (const name of SUB_WIDGET_NAMES) {
            const widget = node._dazzleSubWidgets ? node._dazzleSubWidgets[name] : null;
            if (!widget) continue;

            const idx = node.widgets.indexOf(widget);
            if (idx !== -1) {
                node._dazzleSubWidgetValues[name] = widget.value;
                toRemove.push({ name, widget, idx });
            }
        }

        toRemove.sort((a, b) => b.idx - a.idx);
        for (const item of toRemove) {
            node.widgets.splice(item.idx, 1);
        }
    }

    // Force redraw and recompute size so layout matches the new widget set.
    node.setDirtyCanvas(true, true);
    if (node.computeSize) {
        const newSize = node.computeSize();
        const currentSize = node.size;
        node.setSize([Math.max(currentSize[0], newSize[0]), newSize[1]]);
    }
}


app.registerExtension({
    name: "DazzleKSampler.NoiseAllSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "DazzleKSampler") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Cache references to the three sub-widgets and remember their
            // original `type` so we can restore it on show. ComfyUI's combo
            // widgets default to type "combo" but we read it dynamically in
            // case anything has subclassed it.
            this._dazzleSubWidgets = {};
            this._dazzleSubWidgetValues = {};
            for (const name of SUB_WIDGET_NAMES) {
                const w = findWidget(this, name);
                if (w) {
                    w.origType = w.type;
                    this._dazzleSubWidgets[name] = w;
                    this._dazzleSubWidgetValues[name] = w.value;
                }
            }

            // Hook noise_all's callback so any change triggers a visibility
            // update. Preserve any prior callback. Capture node in closure --
            // ComfyUI invokes widget callbacks with the widget as `this`.
            const node = this;
            const noiseAllWidget = findWidget(this, "noise_all");
            if (noiseAllWidget) {
                const prevCallback = noiseAllWidget.callback;
                noiseAllWidget.callback = function(value) {
                    const ret = prevCallback ? prevCallback.apply(this, arguments) : undefined;
                    updateNoiseAllVisibility(node);
                    return ret;
                };
            }

            // Initial visibility pass deferred so configure() (saved-workflow
            // restore) gets to set noise_all's value first.
            setTimeout(() => updateNoiseAllVisibility(node), 50);

            return r;
        };

        // Also re-run after configure() so saved workflows apply correctly.
        const onConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const node = this;
            setTimeout(() => updateNoiseAllVisibility(node), 50);
            return r;
        };
    },
});
