let sign = "";
let t = eval("{{ timestamp }}") / 1e9;
if (t < 0) {
    t = -t;
    sign = "-";
}
let hours = Math.floor(t / 3600);
let minutes = Math.floor((t % 3600) / 60);
let seconds = t % 60;
return sign + hours + ":" + minutes + ":" + seconds.toFixed(3);