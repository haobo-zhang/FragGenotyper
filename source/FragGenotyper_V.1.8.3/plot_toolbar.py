"""Keep Matplotlib navigation icons readable after a live Qt theme change."""
from PyQt6.QtCore import QEvent
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT


class PlotNavigationToolbar(NavigationToolbar2QT):
    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        }:
            # Matplotlib colors these icons only when constructing them. Build
            # fresh icons against the new palette without touching the figure.
            actions = getattr(self, "_actions", {})
            for _, _, image_file, callback in self.toolitems:
                if image_file and callback in actions:
                    actions[callback].setIcon(self._icon(image_file + ".png"))
