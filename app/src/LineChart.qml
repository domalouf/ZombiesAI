import QtQuick

// One series as a line over its own range, with the last value marked. Drawn on a Canvas: a few hundred
// points, repainted only when the data or the size changes.
Item {
    id: chart
    property var xs: []
    property var ys: []
    property color lineColor: ui.accent
    property string label: ""
    property string valueText: ""
    implicitHeight: 120

    onXsChanged: canvas.requestPaint()
    onYsChanged: canvas.requestPaint()
    onWidthChanged: canvas.requestPaint()
    onHeightChanged: canvas.requestPaint()
    onLineColorChanged: canvas.requestPaint()

    function range(values) {
        let lo = Infinity, hi = -Infinity
        for (const v of values) if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
        if (!isFinite(lo)) return [0, 1]
        if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5 }
        return [lo, hi]
    }
    function fmt(v) {
        const a = Math.abs(v)
        return a >= 1000 ? Math.round(v).toLocaleString(Qt.locale(), 'f', 0)
             : a >= 10 ? v.toLocaleString(Qt.locale(), 'f', 1) : v.toLocaleString(Qt.locale(), 'f', 3)
    }

    Text { id: title; text: chart.label; color: ui.dim; font.pixelSize: 11; elide: Text.ElideRight; width: parent.width - value.width - 8 }
    Text { id: value; anchors.right: parent.right; text: chart.valueText; color: ui.ink; font.pixelSize: 12; font.bold: true }

    Canvas {
        id: canvas
        anchors.fill: parent
        anchors.topMargin: 18
        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            const n = Math.min(chart.xs.length, chart.ys.length)
            const w = width, h = height, padL = 46, padB = 4
            ctx.strokeStyle = ui.line
            ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(padL, h - padB + 0.5); ctx.lineTo(w, h - padB + 0.5); ctx.stroke()
            if (n < 2) {
                ctx.fillStyle = ui.faint
                ctx.font = "11px sans-serif"
                ctx.fillText(qsTr("not enough points yet"), padL + 8, h / 2)
                return
            }
            const [x0, x1] = chart.range(chart.xs), [y0, y1] = chart.range(chart.ys)
            const px = x => padL + (w - padL - 6) * (x - x0) / (x1 - x0 || 1)
            const py = y => 4 + (h - padB - 8) * (1 - (y - y0) / (y1 - y0))
            ctx.fillStyle = ui.faint
            ctx.font = "10px sans-serif"
            ctx.fillText(chart.fmt(y1), 0, 12)
            ctx.fillText(chart.fmt(y0), 0, h - padB - 2)
            ctx.strokeStyle = chart.lineColor
            ctx.lineWidth = 1.6
            ctx.beginPath()
            let started = false, lx = 0, ly = 0
            for (let i = 0; i < n; ++i) {
                const y = chart.ys[i]
                if (y === null || !isFinite(y)) continue
                lx = px(chart.xs[i]); ly = py(y)
                if (!started) { ctx.moveTo(lx, ly); started = true } else ctx.lineTo(lx, ly)
            }
            ctx.stroke()
            ctx.fillStyle = chart.lineColor
            ctx.beginPath(); ctx.arc(lx, ly, 3, 0, 2 * Math.PI); ctx.fill()
        }
    }
}
