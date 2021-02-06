(
s.freeAll;
s.quit;
~additiveControlDef = SynthDef(\additiveControl, {
	var trig, amp, freq, stretch;

	amp = MouseY.kr();
	freq = MouseX.kr(55, 1760, 1);
	stretch = amp / 1000;
	trig = Impulse.kr(256);

	SendReply.kr(trig, '/controls/amp', amp);
	SendReply.kr(trig, '/controls/freq', freq);
	SendReply.kr(trig, '/controls/stretch', stretch);
});

// OSC Server address and port
~zak = NetAddr("127.0.0.1", 8000);
// -------------------------- Chords ------------------------------------
OSCdef(\on_amp, { |msg|
	~zak.sendMsg("/controls/amp", msg[3]);
}, '/controls/amp');

OSCdef(\on_freq, { |msg|
	~zak.sendMsg("/controls/freq", msg[3]);
}, '/controls/freq');

OSCdef(\on_stretch, { |msg|
	~zak.sendMsg("/controls/stretch", msg[3]);
}, '/controls/stretch');

s.boot;
s.waitForBoot{
	// Add SynthDefs to the server
	~additiveControlDef.add;
	s.sync;

	~additiveControl = Synth(\additiveControl);

};
)