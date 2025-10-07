const i = special_vars.index;
const glyph = special_vars.glyph_view;
const height = glyph._y2[i] - glyph._y1[i];
return height.toFixed(2) + "{{ unit }}";