let i = slider.value
source.data["x"] = source.data[i + "_" + x];
source.data["y"] = source.data[i + "_" + y];
source.change.emit();