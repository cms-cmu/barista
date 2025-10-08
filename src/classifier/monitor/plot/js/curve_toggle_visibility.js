let visible = true;
for (const t in toggles) {
    visible = visible && toggles[t].active;
}
for (const c in curves) {
    curves[c].visible = visible;
}