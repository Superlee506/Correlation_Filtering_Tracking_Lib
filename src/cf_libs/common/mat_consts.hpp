//
// Created by super on 18-7-16.
//

#ifndef MAT_CONSTS_H_
#define MAT_CONSTS_H_

namespace mat_consts
{
    template<class T>
    struct constants
    {
        const static T c0_5;
        const static T c2_0;
    };

    template<class T> const T constants<T>::c0_5 = static_cast<T>(0.5);
    template<class T> const T constants<T>::c2_0 = static_cast<T>(2.0);
}

#endif
