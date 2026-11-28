import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Start a run: PPO on the fleet (scripts/train_rl.py) or behavioural cloning from demos (scripts/train_bc.py),
// with every setting the trainer has, and presets to save and reload a set of them.
ScrollView {
    id: page
    contentWidth: availableWidth
    clip: true

    property int mode: 0  // 0: RL run, 1: BC training
    readonly property string kind: mode === 0 ? "rl" : "bc"
    readonly property var trees: backend.launch.trees || []
    readonly property var tree: trees[treeBox.currentIndex] || null
    readonly property var checkpoints: tree ? tree.checkpoints || [] : []
    readonly property var presets: backend.presets.filter(p => p.kind === kind)
    property string loadedPreset: ""
    readonly property var bcDemos: backend.bc.demos || []
    property var bcPicked: ({})  // demo name -> false when left out; everything is in by default

    function primaryAction() { start() }

    function rlRequest() {
        return {
            tree: tree ? tree.label : "",
            init: initBox.currentIndex <= 0 ? "fresh" : checkpoints[initBox.currentIndex - 1].path,
            actors: actors.value, name: runName.text.trim(), env: envBox.currentIndex === 1 ? "synthetic" : "real",
            stop_after_min: stopAfter.value, extra: extra.text.trim(),
            settings: backend.buildSettings(rlForm.schema, rlForm.values)
        }
    }
    function bcRequest() {
        const demos = bcDemos.filter(d => bcPicked[d] !== false)
        return {
            name: bcName.text.trim(), demos: demos.length === bcDemos.length ? [] : demos,
            settings: backend.buildSettings(bcForm.schema, bcForm.values), extra: bcExtra.text.trim()
        }
    }
    function start() {
        const body = mode === 0 ? rlRequest() : bcRequest()
        if (loadedPreset !== "")
            body.preset = loadedPreset
        backend.post(mode === 0 ? "/api/start" : "/api/bc", body, mode === 0 ? "start" : "bc")
    }
    function loadPreset(preset) {
        const r = preset.request || {}
        loadedPreset = preset.name
        if (preset.kind === "rl") {
            const t = trees.findIndex(x => x.label === r.tree)
            if (t >= 0) treeBox.currentIndex = t
            const c = checkpoints.findIndex(x => x.path === r.init)
            initBox.currentIndex = r.init === "fresh" ? 0 : c >= 0 ? c + 1 : initBox.currentIndex
            if (r.actors) actors.value = r.actors
            envBox.currentIndex = r.env === "synthetic" ? 1 : 0
            stopAfter.value = r.stop_after_min || 0
            extra.text = r.extra || ""
            rlForm.values = Object.assign({}, r.settings || {})
            rlForm.generation++
        } else {
            const picked = {}
            if ((r.demos || []).length)
                for (const d of bcDemos) picked[d] = r.demos.indexOf(d) >= 0
            bcPicked = picked
            bcExtra.text = r.extra || ""
            bcForm.values = Object.assign({}, r.settings || {})
            bcForm.generation++
        }
        win.toast(qsTr("Loaded preset %1").arg(preset.name), false)
    }
    function resetForm() {
        loadedPreset = ""
        if (mode === 0) { rlForm.values = {}; rlForm.generation++; extra.text = ""; stopAfter.value = 0 }
        else { bcForm.values = {}; bcForm.generation++; bcPicked = {}; bcExtra.text = "" }
    }

    // A run's length at the speed the last real run managed, so "2,000,000 steps" means something.
    function estimate() {
        const speed = tree && tree.speeds && tree.speeds.real
        const steps = Number(rlForm.values.total_steps !== undefined ? rlForm.values.total_steps
                             : (rlForm.schema.find(s => s.name === "total_steps") || {})["default"])
        if (!speed || !speed.per_game || !steps)
            return ""
        const sps = speed.per_game * actors.value
        let text = qsTr("≈ %1 for %2 steps at %3 steps/s (from %4)")
            .arg(backend.formatDuration(steps / sps)).arg(steps.toLocaleString(Qt.locale(), 'f', 0))
            .arg(Math.round(sps)).arg(speed.run)
        if (stopAfter.value > 0)
            text += qsTr(", stopped after %1 min").arg(stopAfter.value)
        return text
    }

    Connections {
        target: backend
        function onActionFinished(tag, ok, response) {
            if (ok && (tag === "start" || tag === "bc")) {
                win.toast(qsTr("Started %1").arg(response.name || response.run || ""), false)
                win.go(2)
            }
            if (ok && tag === "preset-save")
                win.toast(qsTr("Saved the preset"), false)
        }
    }
    // A fresh name each time the server says the last one was taken.
    Connections {
        target: backend
        function onLaunchChanged() { if (!runName.activeFocus && page.tree) runName.text = page.tree.next }
        function onBcChanged() { if (!bcName.activeFocus) bcName.text = backend.bc.next || "" }
    }

    ColumnLayout {
        width: page.availableWidth
        spacing: 16

        Item { Layout.preferredHeight: 8 }
        RowLayout {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Text { text: qsTr("New run"); color: ui.ink; font.pixelSize: 24; font.bold: true; Layout.fillWidth: true }
            TabBar {
                id: tabs
                currentIndex: page.mode
                onCurrentIndexChanged: { page.mode = currentIndex; page.loadedPreset = "" }
                TabButton { Material.roundedScale: theme.radius; text: qsTr("RL on the fleet"); width: implicitWidth + 24 }
                TabButton { Material.roundedScale: theme.radius; text: qsTr("BC from demos"); width: implicitWidth + 24 }
            }
            AccentButton {
                text: page.mode === 0 ? qsTr("Start run") : qsTr("Start training")
                enabled: backend.connected
                onClicked: page.start()
            }
        }

        // Presets: the same row for either kind.
        Panel {
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            title: qsTr("Presets")
            subtitle: page.loadedPreset !== "" ? qsTr("Using %1; a run started now is marked with it").arg(page.loadedPreset)
                                               : qsTr("Save the settings below under a name, and load them again later")
            RowLayout {
                Layout.fillWidth: true
                ComboBox { Material.roundedScale: theme.radius;
                    id: presetBox
                    Layout.preferredWidth: 260
                    model: page.presets.map(p => p.name)
                    displayText: count ? currentText : qsTr("No presets yet")
                    enabled: count > 0
                }
                Button { Material.roundedScale: theme.radius; text: qsTr("Load"); enabled: presetBox.count > 0; onClicked: page.loadPreset(page.presets[presetBox.currentIndex]) }
                Button { Material.roundedScale: theme.radius;
                    text: qsTr("Delete"); flat: true; enabled: presetBox.count > 0
                    onClicked: backend.post("/api/presets/delete", { name: presetBox.currentText }, "preset-delete")
                }
                Item { Layout.fillWidth: true }
                TextField { Material.roundedScale: theme.radius; id: presetName; Layout.preferredWidth: 200; placeholderText: qsTr("Preset name"); text: page.loadedPreset }
                Button { Material.roundedScale: theme.radius;
                    text: qsTr("Save as preset")
                    enabled: presetName.text.trim() !== ""
                    onClicked: {
                        const request = page.mode === 0 ? page.rlRequest() : page.bcRequest()
                        delete request.name  // a preset is reused; the run's name is fresh each time
                        backend.post("/api/presets", { name: presetName.text.trim(), kind: page.kind, request: request }, "preset-save")
                        page.loadedPreset = presetName.text.trim()
                    }
                }
                Button { Material.roundedScale: theme.radius; text: qsTr("Reset"); flat: true; onClicked: page.resetForm() }
            }
        }

        // ---------------------------------------------------------------- RL
        RowLayout {
            visible: page.mode === 0
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            spacing: 16

            Panel {
                Layout.preferredWidth: 420
                Layout.maximumWidth: 460
                Layout.alignment: Qt.AlignTop
                title: qsTr("The run")
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 12
                    Text { text: qsTr("Checkout"); color: ui.dim }
                    ComboBox { Material.roundedScale: theme.radius;
                        id: treeBox
                        Layout.fillWidth: true
                        model: page.trees.map(t => t.label)
                        Component.onCompleted: currentIndex = Math.max(0, page.trees.findIndex(t => t.label === backend.launch["default"]))
                    }
                    Text { text: qsTr("Start from"); color: ui.dim }
                    ComboBox { Material.roundedScale: theme.radius;
                        id: initBox
                        Layout.fillWidth: true
                        model: [qsTr("fresh (no prior)")].concat(page.checkpoints.map(c => "%1 · %2".arg(c.run).arg(c.kind)))
                        currentIndex: page.checkpoints.length ? 1 : 0
                    }
                    Text { text: qsTr("Name"); color: ui.dim }
                    TextField { Material.roundedScale: theme.radius; id: runName; Layout.fillWidth: true; text: page.tree ? page.tree.next : "" }
                    Text { text: qsTr("Games"); color: ui.dim }
                    SpinBox { Material.roundedScale: theme.radius; id: actors; from: 1; to: 16; editable: true; value: page.tree && page.tree.fleet ? page.tree.fleet.n : 7 }
                    Text { text: qsTr("Environment"); color: ui.dim }
                    ComboBox { Material.roundedScale: theme.radius; id: envBox; Layout.fillWidth: true; model: [qsTr("real game"), qsTr("synthetic rehearsal")] }
                    Text { text: qsTr("Stop after (min)"); color: ui.dim }
                    SpinBox { Material.roundedScale: theme.radius; id: stopAfter; from: 0; to: 10080; stepSize: 30; editable: true; value: 0
                              textFromValue: (v, l) => v === 0 ? qsTr("no limit") : Number(v).toLocaleString(l, 'f', 0) }
                    Text { text: qsTr("More flags"); color: ui.dim }
                    TextField { Material.roundedScale: theme.radius; id: extra; Layout.fillWidth: true; placeholderText: "--flag value" }
                }
                Text {
                    Layout.fillWidth: true
                    text: page.estimate()
                    color: ui.dim; font.pixelSize: 12; wrapMode: Text.Wrap
                    visible: text !== ""
                }
                Repeater {
                    model: {
                        const out = []
                        if (page.tree && (page.tree.busy || []).length)
                            out.push(qsTr("%1 is already playing this fleet's games.").arg(page.tree.busy.join(", ")))
                        if (envBox.currentIndex === 0 && page.tree && page.tree.fleet && page.tree.fleet.running < actors.value)
                            out.push(qsTr("Only %1 of %2 games are up: start them under Games first.").arg(page.tree.fleet.running).arg(actors.value))
                        return out
                    }
                    delegate: Text {
                        required property string modelData
                        Layout.fillWidth: true
                        text: "⚠ " + modelData; color: ui.warn; font.pixelSize: 12; wrapMode: Text.Wrap
                    }
                }
            }

            Panel {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                title: qsTr("Settings")
                subtitle: qsTr("scripts/train_rl.py's own options; only the ones you change become flags")
                SettingsForm {
                    id: rlForm
                    Layout.fillWidth: true
                    schema: page.tree ? page.tree.settings || [] : []
                }
            }
        }

        // ---------------------------------------------------------------- BC
        RowLayout {
            visible: page.mode === 1
            Layout.leftMargin: 24; Layout.rightMargin: 24
            Layout.fillWidth: true
            spacing: 16

            Panel {
                Layout.preferredWidth: 420
                Layout.maximumWidth: 460
                Layout.alignment: Qt.AlignTop
                title: qsTr("Training")
                subtitle: qsTr("A new cloned policy from the demos you pick; the next RL run can start from it")
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 12
                    Text { text: qsTr("Name"); color: ui.dim }
                    TextField { Material.roundedScale: theme.radius; id: bcName; Layout.fillWidth: true; text: backend.bc.next || "" }
                    Text { text: qsTr("More flags"); color: ui.dim }
                    TextField { Material.roundedScale: theme.radius; id: bcExtra; Layout.fillWidth: true; placeholderText: "--flag value" }
                }
                RowLayout {
                    Text {
                        Layout.fillWidth: true
                        color: ui.ink; font.bold: true
                        text: qsTr("Demos (%1 of %2)").arg(page.bcDemos.filter(d => page.bcPicked[d] !== false).length).arg(page.bcDemos.length)
                    }
                    Button { Material.roundedScale: theme.radius; flat: true; text: qsTr("All"); onClicked: page.bcPicked = {} }
                    Button { Material.roundedScale: theme.radius; flat: true; text: qsTr("None"); onClicked: { const p = {}; for (const d of page.bcDemos) p[d] = false; page.bcPicked = p } }
                }
                Repeater {
                    model: page.bcDemos
                    delegate: CheckBox { Material.roundedScale: theme.radius;
                        required property string modelData
                        readonly property var info: (backend.demos.demos || []).find(d => d.name === modelData)
                        text: info ? qsTr("%1 · %2 min · %3").arg(modelData).arg(Math.round(info.play_minutes)).arg(info.notes || "") : modelData
                        checked: page.bcPicked[modelData] !== false
                        onToggled: { const p = Object.assign({}, page.bcPicked); p[modelData] = checked; page.bcPicked = p }
                    }
                }
            }

            Panel {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                title: qsTr("Settings")
                subtitle: qsTr("scripts/train_bc.py's own options")
                SettingsForm {
                    id: bcForm
                    Layout.fillWidth: true
                    schema: backend.bc.settings || []
                }
            }
        }
        Item { Layout.preferredHeight: 16 }
    }
}
