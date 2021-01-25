s.options.memSize_(2.pow(20));
s.boot;

// Control rate is 750 Hz (once every 64 samples at 48kHz sampling rate).

(
SynthDef.new(\sine, {
	arg freq=440, atk=0.005, rel=0.3, amp=1, pan=0, vel=0.5, conf=0.0;
	var sig, env;
	// env = Select.kr(conf > 0.3, [0, 1]);
	// conf.poll(1);
	amp = Lag.kr(amp, 0.5);
	sig = SinOsc.ar(freq);
	// env = EnvGen.kr(Env.new([0,1,0],[atk,rel],[1,-1]),doneAction:2) * vel;
	sig = Pan2.ar(sig, pan, amp);
	// sig = sig * env;
	Out.ar(0, sig);
}).add;
)

(
~sin = Synth(\sine);

OSCdef.new(\crepe_freq, { |msg|
	~sin.set(\freq, msg[1]);
}, "/crepe/freq");

OSCdef.new(\crepe_confidence, { |msg|
	~sin.set(\conf, msg[1]);
}, "/crepe/confidence");

OSCdef.new(\crepe_amp, { |msg|
	~sin.set(\amp, msg[1]);
}, "/crepe/amp");

)
