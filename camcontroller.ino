/*
  Reading a serial ASCII-encoded string.

This sketch demonstrates the Serial parseInt() function.
  It looks for an ASCII string of comma-separated values.
  It parses them into ints, and uses those

*/
#include <Stepper.h>

const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution
// for your motor

Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

void setup() {
  // initialize serial:
  Serial.begin(9600);
  // set the speed at 60 rpm:
  myStepper.setSpeed(60);

}

void loop()
{
  	// if there's any serial available, read it:
 	while (Serial.available() > 0)
	{

		// look for the next valid integer in the incoming serial stream:
		int steps = Serial.parseInt();


		// do it again:
		int dir = Serial.parseInt();



		// look for the newline. That's the end of your sentence:
		if (Serial.read() == '\n')
		{
		  	// constrain the values to 0 - 255 and invert
		  	steps = constrain(steps, 0, 255);


			if(dir == 0)
			{

				myStepper.step(steps);
	  		delay(10);
			}

			else if(dir == 1)
			{
        Serial.println("Spinning...1");
				myStepper.step(-steps);
	  			delay(10);
			}

    	}
	}
}
