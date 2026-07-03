// Chasm chat UI helper.
//
// Keeps the Live Chat message panel pinned to the newest messages at the
// bottom, the way a chat client should open. Plain browser JS only — no
// framework, no bundler, no build step. Safe to call repeatedly, so it stays
// compatible with HTMX-style partial swaps if those are wired up later.
(function () {
    "use strict";

    // Scroll every visible message panel to its last message.
    function pinMessagesToBottom() {
        var panels = document.querySelectorAll(".message-scroll");
        for (var i = 0; i < panels.length; i++) {
            var panel = panels[i];
            panel.scrollTop = panel.scrollHeight;
        }
    }

    // Expose for manual hooks / future scripts that mutate the message list.
    window.chasmPinMessagesToBottom = pinMessagesToBottom;

    function pinNow() {
        pinMessagesToBottom();
        // Re-pin on the next frame: scrollHeight can still grow after the first
        // layout pass (wrapped text, late-loading fonts), which would otherwise
        // leave us a few pixels short of the true bottom.
        if (typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(pinMessagesToBottom);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", pinNow);
    } else {
        pinNow();
    }

    // Fonts/images can change message heights after full load; pin once more.
    window.addEventListener("load", pinMessagesToBottom);

    // HTMX (if/when wired up) swaps message partials in place — re-pin after
    // the new content is inserted and has settled. These events simply never
    // fire when HTMX is absent, so this is harmless today.
    document.addEventListener("htmx:afterSwap", pinMessagesToBottom);
    document.addEventListener("htmx:afterSettle", pinMessagesToBottom);
})();

// Per-message injection panel. Clicking a message row swaps the right-hand prompt
// panel from the default "next prompt" view to THAT message's injected
// lore/quest/action entries + the actions it carried. "Back to next prompt"
// restores the default. Rows and detail sections are paired by list position
// (data-prompt-row / data-msg-detail) so the mapping is exact even when message
// ids repeat across segments. Plain browser JS; uses event delegation on the
// document so it keeps working after partial swaps re-render the message list.
(function () {
    "use strict";

    function showNextPrompt() {
        var nextView = document.getElementById("prompt-next-view");
        var msgView = document.getElementById("prompt-message-view");
        if (nextView) {
            nextView.hidden = false;
        }
        if (msgView) {
            msgView.hidden = true;
            var details = msgView.querySelectorAll("[data-msg-detail]");
            for (var i = 0; i < details.length; i++) {
                details[i].hidden = true;
            }
        }
        var rows = document.querySelectorAll(".message-row.is-active");
        for (var j = 0; j < rows.length; j++) {
            rows[j].classList.remove("is-active");
        }
    }

    function showMessageDetail(key, row) {
        var nextView = document.getElementById("prompt-next-view");
        var msgView = document.getElementById("prompt-message-view");
        if (!msgView) {
            return;
        }
        var target = msgView.querySelector('[data-msg-detail="' + key + '"]');
        if (!target) {
            return;
        }
        var details = msgView.querySelectorAll("[data-msg-detail]");
        for (var i = 0; i < details.length; i++) {
            details[i].hidden = details[i] !== target;
        }
        if (nextView) {
            nextView.hidden = true;
        }
        msgView.hidden = false;
        msgView.scrollTop = 0;

        var rows = document.querySelectorAll(".message-row.is-active");
        for (var j = 0; j < rows.length; j++) {
            rows[j].classList.remove("is-active");
        }
        if (row) {
            row.classList.add("is-active");
        }
    }

    // Delegate clicks: a message row opens its detail; the back button restores.
    document.addEventListener("click", function (event) {
        var back = event.target.closest ? event.target.closest("#prompt-back") : null;
        if (back) {
            showNextPrompt();
            return;
        }
        var row = event.target.closest ? event.target.closest("[data-prompt-row]") : null;
        if (row) {
            showMessageDetail(row.getAttribute("data-prompt-row"), row);
        }
    });

    // Keyboard parity: Enter / Space on a focused row opens its detail.
    document.addEventListener("keydown", function (event) {
        if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") {
            return;
        }
        var row = event.target.closest ? event.target.closest("[data-prompt-row]") : null;
        if (row) {
            event.preventDefault();
            showMessageDetail(row.getAttribute("data-prompt-row"), row);
        }
    });
})();

// Settings page: keep the Voice Cloning panel in sync with the selected local
// engine, and let each cloned character be tested with a one-off generation.
(function () {
    "use strict";

    var TEST_TEXT = "Hello there, this is a quick test of my cloned voice.";

    // Swap the voice-clone panel when the local engine changes. Clone status is
    // per-engine, so the panel must reflect the *selected* engine rather than the
    // last-saved one (otherwise every engine looks cloned).
    function wireEngineSwap() {
        var select = document.querySelector('select[name="local_engine"]');
        var panel = document.getElementById("voice-clone-panel");
        if (!select || !panel) {
            return;
        }
        select.addEventListener("change", function () {
            var url = "/partials/settings/voice-clone?engine=" + encodeURIComponent(select.value);
            fetch(url, { headers: { Accept: "text/html" } })
                .then(function (response) {
                    return response.ok ? response.text() : Promise.reject(response.status);
                })
                .then(function (html) {
                    panel.innerHTML = html;
                })
                .catch(function () {
                    // Leave the current panel in place if the fetch fails.
                });
        });
    }

    // Generate a fresh test clip for one character, play it, then revoke the
    // blob URL on end/error so nothing accumulates in memory.
    function runVoiceTest(button) {
        var character = button.getAttribute("data-character");
        if (!character || button.disabled) {
            return;
        }
        var label = button.textContent;
        button.disabled = true;
        button.textContent = "Testing…";

        function reset() {
            button.disabled = false;
            button.textContent = label;
        }

        fetch("/api/headless/v1/speech/synthesize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: TEST_TEXT, characterName: character })
        })
            .then(function (response) {
                return response.ok ? response.json() : Promise.reject(response.status);
            })
            .then(function (data) {
                var base64 = data && data.audio && data.audio.data;
                if (!base64) {
                    return Promise.reject("no audio");
                }
                var binary = atob(base64);
                var bytes = new Uint8Array(binary.length);
                for (var i = 0; i < binary.length; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }
                var blob = new Blob([bytes], { type: data.mimeType || "audio/wav" });
                var objectUrl = URL.createObjectURL(blob);
                var audio = new Audio(objectUrl);
                function cleanup() {
                    URL.revokeObjectURL(objectUrl);
                    reset();
                }
                audio.addEventListener("ended", cleanup);
                audio.addEventListener("error", cleanup);
                return audio.play().catch(cleanup);
            })
            .catch(function () {
                button.textContent = "Test failed";
                setTimeout(reset, 1500);
            });
    }

    // Collect the live TTS-tuning form values as the JSON the worker expects.
    // Sends every [data-tuning] control by name (minus the "tuning_" prefix), so
    // the values shown right now win over the saved settings for this one test.
    function collectTuning() {
        var tuning = {};
        var inputs = document.querySelectorAll("[data-tuning]");
        for (var i = 0; i < inputs.length; i++) {
            var input = inputs[i];
            var key = input.name.replace(/^tuning_/, "");
            var value = parseFloat(input.value);
            if (key && !isNaN(value)) {
                tuning[key] = value;
            }
        }
        return tuning;
    }

    // Play a fresh clip generated with the CURRENT tuning-form values, on the
    // chosen cloned voice — the live tune→test loop. Mirrors runVoiceTest but
    // pulls the character from the tuning card's picker and attaches `tuning`.
    function runTuningTest(button) {
        if (button.disabled) {
            return;
        }
        var picker = document.getElementById("tuning_test_voice");
        var character = picker && picker.value;
        var status = document.querySelector(".tuning-test-status");
        if (!character) {
            return;
        }
        var label = button.textContent;
        button.disabled = true;
        button.textContent = "Testing…";
        if (status) {
            status.hidden = true;
        }

        function reset() {
            button.disabled = false;
            button.textContent = label;
        }
        function fail(message) {
            if (status) {
                status.textContent = message;
                status.hidden = false;
            }
            button.textContent = "Test failed";
            setTimeout(reset, 1500);
        }

        fetch("/api/headless/v1/speech/synthesize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: TEST_TEXT,
                characterName: character,
                tuning: collectTuning()
            })
        })
            .then(function (response) {
                return response.ok ? response.json() : Promise.reject(response.status);
            })
            .then(function (data) {
                var base64 = data && data.audio && data.audio.data;
                if (!base64) {
                    return Promise.reject("no audio");
                }
                var binary = atob(base64);
                var bytes = new Uint8Array(binary.length);
                for (var i = 0; i < binary.length; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }
                var blob = new Blob([bytes], { type: data.mimeType || "audio/wav" });
                var objectUrl = URL.createObjectURL(blob);
                var audio = new Audio(objectUrl);
                function cleanup() {
                    URL.revokeObjectURL(objectUrl);
                    reset();
                }
                audio.addEventListener("ended", cleanup);
                audio.addEventListener("error", cleanup);
                return audio.play().catch(cleanup);
            })
            .catch(function () {
                fail("Test failed (is the TTS worker running?).");
            });
    }

    // Delegate clicks so test buttons keep working after the panel is swapped.
    document.addEventListener("click", function (event) {
        if (!event.target.closest) {
            return;
        }
        var tuneBtn = event.target.closest(".tuning-test-btn");
        if (tuneBtn) {
            event.preventDefault();
            runTuningTest(tuneBtn);
            return;
        }
        var button = event.target.closest(".voice-test-btn");
        if (button) {
            event.preventDefault();
            runVoiceTest(button);
        }
    });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", wireEngineSwap);
    } else {
        wireEngineSwap();
    }
})();

// Connection indicator (chat rail): chasm is a passive backend, so instead of a
// Play button the rail shows whether the in-game plugin is talking to us. The
// plugin writes a heartbeat file while running; /connection/status reports its
// freshness. Poll every ~1.5s and toggle the dot/label. Dependency-free.
(function () {
    "use strict";

    var POLL_MS = 1500;

    function wireConnection() {
        var indicator = document.getElementById("connection-indicator");
        var label = document.getElementById("connection-label");
        if (!indicator || !label) {
            return;
        }

        // Map the lifecycle phase from /connection/status to the rail's dot +
        // label. `starting` shows a warming-up state (llama.cpp ~12s, TTS model
        // ~45s); `connected` is the steady green; everything else reads as offline.
        function render(phase, connected) {
            var isConnected = phase === "connected" || (!phase && connected);
            var isStarting = phase === "starting" || phase === "stopping";
            indicator.classList.toggle("is-connected", isConnected);
            indicator.classList.toggle("is-starting", isStarting);
            var labelText;
            if (phase === "starting") {
                labelText = "Starting…";
            } else if (phase === "stopping") {
                labelText = "Stopping…";
            } else if (isConnected) {
                labelText = "Connected";
            } else {
                labelText = "Not connected";
            }
            label.textContent = labelText;
            indicator.setAttribute("aria-label", "Game connection: " + labelText);
        }

        function poll() {
            fetch("/connection/status", { headers: { Accept: "application/json" } })
                .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
                .then(function (data) {
                    render(data && data.phase, !!(data && data.connected));
                })
                .catch(function () { render("disconnected", false); });
        }

        poll();
        setInterval(poll, POLL_MS);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", wireConnection);
    } else {
        wireConnection();
    }
})();

// Settings > Interface: keep the accent colour picker, its readout, and the
// preset swatches in sync; Settings > Profiles: activate a profile via the
// existing /profile/select endpoint, then reload.
(function () {
    "use strict";

    function wireInterface() {
        var color = document.getElementById("accent");
        var readout = document.getElementById("accent-readout");
        if (color) {
            var sync = function () {
                if (readout) {
                    readout.value = color.value;
                }
                document.querySelectorAll(".accent-dot").forEach(function (dot) {
                    var match =
                        (dot.getAttribute("data-accent") || "").toLowerCase() ===
                        color.value.toLowerCase();
                    dot.classList.toggle("is-selected", match);
                });
            };
            color.addEventListener("input", sync);
            document.querySelectorAll(".accent-dot").forEach(function (dot) {
                dot.addEventListener("click", function () {
                    color.value = dot.getAttribute("data-accent") || color.value;
                    sync();
                });
            });
        }
    }

    function wireProfiles() {
        document.querySelectorAll(".profile-activate").forEach(function (button) {
            button.addEventListener("click", function () {
                var id = button.getAttribute("data-profile-id");
                if (!id || button.disabled) {
                    return;
                }
                button.disabled = true;
                button.textContent = "Activating…";
                fetch("/profile/select", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ id: id }),
                })
                    .then(function (res) {
                        if (res.ok) {
                            window.location.reload();
                        } else {
                            button.disabled = false;
                            button.textContent = "Activate";
                        }
                    })
                    .catch(function () {
                        button.disabled = false;
                        button.textContent = "Activate";
                    });
            });
        });
    }

    function wire() {
        wireInterface();
        wireProfiles();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", wire);
    } else {
        wire();
    }
})();

// Lorebook editor: add a blank entry card (uid = max+1) and delete entry cards.
// Deletes are recorded into a hidden `deleted` CSV so the server drops them from
// the original JSON on save (it overlays present uids + applies deletes).
(function () {
    "use strict";

    function initLorebookEditor() {
        var form = document.getElementById("lorebook-form");
        if (!form) {
            return;
        }
        var list = document.getElementById("lorebook-entries");
        var tmpl = document.getElementById("lorebook-entry-template");
        var deletedInput = document.getElementById("lorebook-deleted");
        var countEl = document.getElementById("book-entry-count");
        var deleted = [];

        function nextUid() {
            var max = -1;
            list.querySelectorAll(".book-entry[data-uid]").forEach(function (card) {
                var n = parseInt(card.getAttribute("data-uid"), 10);
                if (!isNaN(n) && n > max) {
                    max = n;
                }
            });
            return max + 1;
        }

        function updateCount() {
            if (countEl) {
                countEl.textContent = String(list.querySelectorAll(".book-entry").length);
            }
        }

        function addEntry() {
            if (!tmpl || !list) {
                return;
            }
            var uid = String(nextUid());
            // Substitute the placeholder uid throughout the cloned markup.
            var html = tmpl.innerHTML.replace(/__UID__/g, uid);
            var holder = document.createElement("div");
            holder.innerHTML = html.trim();
            var card = holder.firstElementChild;
            list.appendChild(card);
            updateCount();
            var firstField = card.querySelector("textarea, input[type=text]");
            if (firstField) {
                firstField.focus();
            }
            card.scrollIntoView({ behavior: "smooth", block: "center" });
        }

        function deleteEntry(card) {
            var uid = card.getAttribute("data-uid");
            var name = (card.querySelector(".book-entry-name") || {}).textContent || "this entry";
            if (!window.confirm("Delete " + name.trim() + "? It will be removed when you Save.")) {
                return;
            }
            if (uid) {
                deleted.push(uid);
                if (deletedInput) {
                    deletedInput.value = deleted.join(",");
                }
            }
            card.parentNode.removeChild(card);
            updateCount();
        }

        var addButtons = [
            document.getElementById("lorebook-add"),
            document.getElementById("lorebook-add-foot"),
        ];
        addButtons.forEach(function (button) {
            if (button) {
                button.addEventListener("click", addEntry);
            }
        });

        // Delegate per-card delete (works for cards added after load too).
        list.addEventListener("click", function (event) {
            var del = event.target.closest(".book-del");
            if (!del) {
                return;
            }
            // Keep the click from toggling the <details> summary.
            event.preventDefault();
            event.stopPropagation();
            var card = del.closest(".book-entry");
            if (card) {
                deleteEntry(card);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initLorebookEditor);
    } else {
        initLorebookEditor();
    }
})();
