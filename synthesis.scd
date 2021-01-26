s.options.memSize_(2.pow(20));
s.boot;

// Control rate is 750 Hz (once every 64 samples at 48kHz sampling rate).

(
SynthDef.new(\sine, {
	arg freq=440, amp=1, pan=0, harmonics=Array.fill(65, 0);
	var sig;

	sig = Mix.ar(SinOsc.ar((1..65) * freq, harmonics));

	sig = Pan2.ar(sig, pan, amp);
	Out.ar(0, sig);
}).add;
)

(
~sin = Synth(\sine);

OSCdef.new(\crepe_freq, { |msg|
	~sin.set(\freq, msg[1] * 2);
}, "/crepe/freq");

OSCdef.new(\crepe_harmonics, { |msg|
	var harmonics = msg.copyToEnd(1);
	~sin.setn(\harmonics, harmonics);
}, "/crepe/harmonics");

OSCdef.new(\crepe_amp, { |msg|
	~sin.set(\amp, msg[1]);
}, "/crepe/amp");

)
