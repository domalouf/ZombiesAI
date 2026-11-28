import QtQuick
import QtQuick.Layouts

// Seven days of recorded minutes as bars, with the daily goal as a line across them.
Item {
    id: bars
    property var week: []
    property real goal: 20
    implicitHeight: 70
    readonly property real ceiling: Math.max(goal * 1.25, ...week.map(d => d.minutes || 0), 1)

    Rectangle {  // the goal
        x: 0; width: parent.width; height: 1
        y: 52 * (1 - bars.goal / bars.ceiling)
        color: ui.faint
    }
    Row {
        anchors.fill: parent
        spacing: 6
        Repeater {
            model: bars.week
            delegate: Item {
                required property var modelData
                required property int index
                width: (bars.width - 6 * 6) / 7
                height: bars.height
                Rectangle {
                    anchors.bottom: day.top
                    anchors.bottomMargin: 4
                    width: parent.width
                    height: Math.max(2, 52 * (modelData.minutes || 0) / bars.ceiling)
                    radius: Math.min(2, ui.radius)
                    color: (modelData.minutes || 0) >= bars.goal ? ui.good : index === bars.week.length - 1 ? ui.ink : ui.faint
                }
                Text {
                    id: day
                    anchors.bottom: parent.bottom
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: Qt.locale().dayName(new Date(modelData.date + "T12:00:00").getDay(), Locale.NarrowFormat)
                    color: ui.dim; font.pixelSize: 10
                }
            }
        }
    }
}
