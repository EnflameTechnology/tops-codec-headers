
/**
 * Copyright 2020 The Enflame Tech Company. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TOPSCODEC_EXPORT_H
#define TOPSCODEC_EXPORT_H

#ifdef TOPSCODEC_STATIC_DEFINE
#  define TOPSCODEC_EXPORT
#  define TOPSCODEC_NO_EXPORT
#else
#  ifndef TOPSCODEC_EXPORT
#    ifdef libtopscodec_EXPORTS
        /* We are building this library */
#      define TOPSCODEC_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define TOPSCODEC_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef TOPSCODEC_NO_EXPORT
#    define TOPSCODEC_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef TOPSCODEC_DEPRECATED
#  define TOPSCODEC_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef TOPSCODEC_DEPRECATED_EXPORT
#  define TOPSCODEC_DEPRECATED_EXPORT TOPSCODEC_EXPORT TOPSCODEC_DEPRECATED
#endif

#ifndef TOPSCODEC_DEPRECATED_NO_EXPORT
#  define TOPSCODEC_DEPRECATED_NO_EXPORT TOPSCODEC_NO_EXPORT TOPSCODEC_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef TOPSCODEC_NO_DEPRECATED
#    define TOPSCODEC_NO_DEPRECATED
#  endif
#endif

#endif /* TOPSCODEC_EXPORT_H */
