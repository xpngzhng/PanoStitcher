#pragma once
#include <cmath>

class Mtx33
{
public:
    /** we define the Matrix3 as 3 colums of 3 rows */
    double m[3][3];

public:

    Mtx33 operator*(const Mtx33& ot) const
    {
        Mtx33 Result;
        Result.m[0][0] = m[0][0] * ot.m[0][0] + m[0][1] * ot.m[1][0] + m[0][2] * ot.m[2][0];
        Result.m[0][1] = m[0][0] * ot.m[0][1] + m[0][1] * ot.m[1][1] + m[0][2] * ot.m[2][1];
        Result.m[0][2] = m[0][0] * ot.m[0][2] + m[0][1] * ot.m[1][2] + m[0][2] * ot.m[2][2];

        Result.m[1][0] = m[1][0] * ot.m[0][0] + m[1][1] * ot.m[1][0] + m[1][2] * ot.m[2][0];
        Result.m[1][1] = m[1][0] * ot.m[0][1] + m[1][1] * ot.m[1][1] + m[1][2] * ot.m[2][1];
        Result.m[1][2] = m[1][0] * ot.m[0][2] + m[1][1] * ot.m[1][2] + m[1][2] * ot.m[2][2];

        Result.m[2][0] = m[2][0] * ot.m[0][0] + m[2][1] * ot.m[1][0] + m[2][2] * ot.m[2][0];
        Result.m[2][1] = m[2][0] * ot.m[0][1] + m[2][1] * ot.m[1][1] + m[2][2] * ot.m[2][1];
        Result.m[2][2] = m[2][0] * ot.m[0][2] + m[2][1] * ot.m[1][2] + m[2][2] * ot.m[2][2];
        return Result;
    }

    /** set rotation in panotools style,
    *  code adapted from Panotools-Script by Bruno Postle
    */
    void SetRotationPT(double yaw, double pitch, double roll)
    {
        double cosr = cos(roll);
        double sinr = sin(roll);
        double cosp = cos(pitch);
        double sinp = sin(0 - pitch);
        double cosy = cos(yaw);
        double siny = sin(0 - yaw);

        Mtx33 rollm;

        /*
        rollm[0][0] = new Math::Matrix ([        1,       0,       0 ],
                                        [        0,   cosr,-1*sinr ],
                                        [        0,   sinr,   cosr ]);
        */

        rollm.m[0][0] = 1.0;      rollm.m[0][1] = 0.0;      rollm.m[0][2] = 0.0;
        rollm.m[1][0] = 0.0;      rollm.m[1][1] = cosr;     rollm.m[1][2] = -sinr;
        rollm.m[2][0] = 0.0;      rollm.m[2][1] = sinr;     rollm.m[2][2] = cosr;

        /*
        my pitchm = new Math::Matrix ([    cosp,       0,   sinp ],
                                      [        0,       1,       0 ],
                                      [ -1*sinp,       0,   cosp ]);
        */

        Mtx33 pitchm;
        pitchm.m[0][0] = cosp;   pitchm.m[0][1] = 0.0;  pitchm.m[0][2] = sinp;
        pitchm.m[1][0] = 0.0;   pitchm.m[1][1] = 1;  pitchm.m[1][2] = 0.0;
        pitchm.m[2][0] = -sinp;  pitchm.m[2][1] = 0.0;  pitchm.m[2][2] = cosp;

        /*
        my yawm   = new Math::Matrix ([    cosy,-1*siny,       0 ],
                                      [    siny,   cosy,       0 ],
                                      [        0,       0,       1 ]);
        */
        Mtx33 yawm;
        yawm.m[0][0] = cosy;   yawm.m[0][1] = -siny;   yawm.m[0][2] = 0.0;
        yawm.m[1][0] = siny;   yawm.m[1][1] = cosy;   yawm.m[1][2] = 0.0;
        yawm.m[2][0] = 0.0;    yawm.m[2][1] = 0.0;   yawm.m[2][2] = 1.0;


        *this = yawm * pitchm * rollm;
    }

    /** GetRotation in panotools style. */
    void GetRotationPT(double & Yaw, double & Pitch, double & Roll)
    {
        /*
        my $matrix = shift;
        my $roll = atan2 ($matrix->[2]->[1], $matrix->[2]->[2]);
        my $pitch = -1 * asin (-1 * $matrix->[2]->[0]);
        my $yaw = atan2 (-1 * $matrix->[1]->[0], $matrix->[0]->[0]);
        return ($roll, $pitch, $yaw);

        */
        Roll = atan2(m[2][1], m[2][2]);
        Pitch = -asin(-m[2][0]);
        Yaw = atan2(-m[1][0], m[0][0]);
    }

};