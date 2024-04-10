// The OpenCL kernel code originates from xmr-stak: https://github.com/fireice-uk/xmr-stak/tree/master/xmrstak/backend.
/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  */

static const __constant ulong keccakf_rndc[24] =
{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

static const __constant uchar sbox[256] =
{
	0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
	0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
	0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
	0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
	0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
	0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
	0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
	0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
	0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
	0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
	0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
	0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
	0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
	0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
	0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
	0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};


#ifndef WOLF_AES_CL
#define WOLF_AES_CL

/* For Mesa clover support */
#ifdef cl_clang_storage_class_specifiers
#   pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

#ifdef cl_amd_media_ops2
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#else
/* taken from: https://www.khronos.org/registry/OpenCL/extensions/amd/cl_amd_media_ops2.txt
 *     Built-in Function:
 *     uintn amd_bfe (uintn src0, uintn src1, uintn src2)
 *   Description
 *     NOTE: operator >> below represent logical right shift
 *     offset = src1.s0 & 31;
 *     width = src2.s0 & 31;
 *     if width = 0
 *         dst.s0 = 0;
 *     else if (offset + width) < 32
 *         dst.s0 = (src0.s0 << (32 - offset - width)) >> (32 - width);
 *     else
 *         dst.s0 = src0.s0 >> offset;
 *     similar operation applied to other components of the vectors
 */
inline int amd_bfe(const uint src0, const uint offset, const uint width)
{
	/* casts are removed because we can implement everything as uint
	 * int offset = src1;
	 * int width = src2;
	 * remove check for edge case, this function is always called with
	 * `width==8`
	 * @code
	 *   if ( width == 0 )
	 *      return 0;
	 * @endcode
	 */
	if ( (offset + width) < 32u )
		return (src0 << (32u - offset - width)) >> (32u - width);

	return src0 >> offset;
}
#endif

// AES table - the other three are generated on the fly

static const __constant uint AES0_C[256] =
{
	0xA56363C6U, 0x847C7CF8U, 0x997777EEU, 0x8D7B7BF6U,
	0x0DF2F2FFU, 0xBD6B6BD6U, 0xB16F6FDEU, 0x54C5C591U,
	0x50303060U, 0x03010102U, 0xA96767CEU, 0x7D2B2B56U,
	0x19FEFEE7U, 0x62D7D7B5U, 0xE6ABAB4DU, 0x9A7676ECU,
	0x45CACA8FU, 0x9D82821FU, 0x40C9C989U, 0x877D7DFAU,
	0x15FAFAEFU, 0xEB5959B2U, 0xC947478EU, 0x0BF0F0FBU,
	0xECADAD41U, 0x67D4D4B3U, 0xFDA2A25FU, 0xEAAFAF45U,
	0xBF9C9C23U, 0xF7A4A453U, 0x967272E4U, 0x5BC0C09BU,
	0xC2B7B775U, 0x1CFDFDE1U, 0xAE93933DU, 0x6A26264CU,
	0x5A36366CU, 0x413F3F7EU, 0x02F7F7F5U, 0x4FCCCC83U,
	0x5C343468U, 0xF4A5A551U, 0x34E5E5D1U, 0x08F1F1F9U,
	0x937171E2U, 0x73D8D8ABU, 0x53313162U, 0x3F15152AU,
	0x0C040408U, 0x52C7C795U, 0x65232346U, 0x5EC3C39DU,
	0x28181830U, 0xA1969637U, 0x0F05050AU, 0xB59A9A2FU,
	0x0907070EU, 0x36121224U, 0x9B80801BU, 0x3DE2E2DFU,
	0x26EBEBCDU, 0x6927274EU, 0xCDB2B27FU, 0x9F7575EAU,
	0x1B090912U, 0x9E83831DU, 0x742C2C58U, 0x2E1A1A34U,
	0x2D1B1B36U, 0xB26E6EDCU, 0xEE5A5AB4U, 0xFBA0A05BU,
	0xF65252A4U, 0x4D3B3B76U, 0x61D6D6B7U, 0xCEB3B37DU,
	0x7B292952U, 0x3EE3E3DDU, 0x712F2F5EU, 0x97848413U,
	0xF55353A6U, 0x68D1D1B9U, 0x00000000U, 0x2CEDEDC1U,
	0x60202040U, 0x1FFCFCE3U, 0xC8B1B179U, 0xED5B5BB6U,
	0xBE6A6AD4U, 0x46CBCB8DU, 0xD9BEBE67U, 0x4B393972U,
	0xDE4A4A94U, 0xD44C4C98U, 0xE85858B0U, 0x4ACFCF85U,
	0x6BD0D0BBU, 0x2AEFEFC5U, 0xE5AAAA4FU, 0x16FBFBEDU,
	0xC5434386U, 0xD74D4D9AU, 0x55333366U, 0x94858511U,
	0xCF45458AU, 0x10F9F9E9U, 0x06020204U, 0x817F7FFEU,
	0xF05050A0U, 0x443C3C78U, 0xBA9F9F25U, 0xE3A8A84BU,
	0xF35151A2U, 0xFEA3A35DU, 0xC0404080U, 0x8A8F8F05U,
	0xAD92923FU, 0xBC9D9D21U, 0x48383870U, 0x04F5F5F1U,
	0xDFBCBC63U, 0xC1B6B677U, 0x75DADAAFU, 0x63212142U,
	0x30101020U, 0x1AFFFFE5U, 0x0EF3F3FDU, 0x6DD2D2BFU,
	0x4CCDCD81U, 0x140C0C18U, 0x35131326U, 0x2FECECC3U,
	0xE15F5FBEU, 0xA2979735U, 0xCC444488U, 0x3917172EU,
	0x57C4C493U, 0xF2A7A755U, 0x827E7EFCU, 0x473D3D7AU,
	0xAC6464C8U, 0xE75D5DBAU, 0x2B191932U, 0x957373E6U,
	0xA06060C0U, 0x98818119U, 0xD14F4F9EU, 0x7FDCDCA3U,
	0x66222244U, 0x7E2A2A54U, 0xAB90903BU, 0x8388880BU,
	0xCA46468CU, 0x29EEEEC7U, 0xD3B8B86BU, 0x3C141428U,
	0x79DEDEA7U, 0xE25E5EBCU, 0x1D0B0B16U, 0x76DBDBADU,
	0x3BE0E0DBU, 0x56323264U, 0x4E3A3A74U, 0x1E0A0A14U,
	0xDB494992U, 0x0A06060CU, 0x6C242448U, 0xE45C5CB8U,
	0x5DC2C29FU, 0x6ED3D3BDU, 0xEFACAC43U, 0xA66262C4U,
	0xA8919139U, 0xA4959531U, 0x37E4E4D3U, 0x8B7979F2U,
	0x32E7E7D5U, 0x43C8C88BU, 0x5937376EU, 0xB76D6DDAU,
	0x8C8D8D01U, 0x64D5D5B1U, 0xD24E4E9CU, 0xE0A9A949U,
	0xB46C6CD8U, 0xFA5656ACU, 0x07F4F4F3U, 0x25EAEACFU,
	0xAF6565CAU, 0x8E7A7AF4U, 0xE9AEAE47U, 0x18080810U,
	0xD5BABA6FU, 0x887878F0U, 0x6F25254AU, 0x722E2E5CU,
	0x241C1C38U, 0xF1A6A657U, 0xC7B4B473U, 0x51C6C697U,
	0x23E8E8CBU, 0x7CDDDDA1U, 0x9C7474E8U, 0x211F1F3EU,
	0xDD4B4B96U, 0xDCBDBD61U, 0x868B8B0DU, 0x858A8A0FU,
	0x907070E0U, 0x423E3E7CU, 0xC4B5B571U, 0xAA6666CCU,
	0xD8484890U, 0x05030306U, 0x01F6F6F7U, 0x120E0E1CU,
	0xA36161C2U, 0x5F35356AU, 0xF95757AEU, 0xD0B9B969U,
	0x91868617U, 0x58C1C199U, 0x271D1D3AU, 0xB99E9E27U,
	0x38E1E1D9U, 0x13F8F8EBU, 0xB398982BU, 0x33111122U,
	0xBB6969D2U, 0x70D9D9A9U, 0x898E8E07U, 0xA7949433U,
	0xB69B9B2DU, 0x221E1E3CU, 0x92878715U, 0x20E9E9C9U,
	0x49CECE87U, 0xFF5555AAU, 0x78282850U, 0x7ADFDFA5U,
	0x8F8C8C03U, 0xF8A1A159U, 0x80898909U, 0x170D0D1AU,
	0xDABFBF65U, 0x31E6E6D7U, 0xC6424284U, 0xB86868D0U,
	0xC3414182U, 0xB0999929U, 0x772D2D5AU, 0x110F0F1EU,
	0xCBB0B07BU, 0xFC5454A8U, 0xD6BBBB6DU, 0x3A16162CU
};

#define BYTE(x, y)	(amd_bfe((x), (y) << 3U, 8U))

uint4 AES_Round(const __local uint *AES0, const __local uint *AES1, const __local uint *AES2, const __local uint *AES3, const uint4 X, uint4 key)
{
	key.s0 ^= AES0[BYTE(X.s0, 0)];
	key.s1 ^= AES0[BYTE(X.s1, 0)];
	key.s2 ^= AES0[BYTE(X.s2, 0)];
	key.s3 ^= AES0[BYTE(X.s3, 0)];

	key.s0 ^= AES2[BYTE(X.s2, 2)];
	key.s1 ^= AES2[BYTE(X.s3, 2)];
	key.s2 ^= AES2[BYTE(X.s0, 2)];
	key.s3 ^= AES2[BYTE(X.s1, 2)];

	key.s0 ^= AES1[BYTE(X.s1, 1)];
	key.s1 ^= AES1[BYTE(X.s2, 1)];
	key.s2 ^= AES1[BYTE(X.s3, 1)];
	key.s3 ^= AES1[BYTE(X.s0, 1)];

	key.s0 ^= AES3[BYTE(X.s3, 3)];
	key.s1 ^= AES3[BYTE(X.s0, 3)];
	key.s2 ^= AES3[BYTE(X.s1, 3)];
	key.s3 ^= AES3[BYTE(X.s2, 3)];

	return key;
}

#endif

static const __constant uint keccakf_rotc[24] =
{
	1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
	27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

static const __constant uint keccakf_piln[24] =
{
	10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
	15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

inline void keccakf1600_1(ulong st[25])
{
	int i, round;
	ulong t, bc[5];

	#pragma unroll 1
	for (round = 0; round < 24; ++round)
	{
		bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20] ^ rotate(st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], 1UL);
		bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21] ^ rotate(st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23], 1UL);
		bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22] ^ rotate(st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24], 1UL);
		bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23] ^ rotate(st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], 1UL);
		bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24] ^ rotate(st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21], 1UL);

		st[0] ^= bc[4];
		st[5] ^= bc[4];
		st[10] ^= bc[4];
		st[15] ^= bc[4];
		st[20] ^= bc[4];

		st[1] ^= bc[0];
		st[6] ^= bc[0];
		st[11] ^= bc[0];
		st[16] ^= bc[0];
		st[21] ^= bc[0];

		st[2] ^= bc[1];
		st[7] ^= bc[1];
		st[12] ^= bc[1];
		st[17] ^= bc[1];
		st[22] ^= bc[1];

		st[3] ^= bc[2];
		st[8] ^= bc[2];
		st[13] ^= bc[2];
		st[18] ^= bc[2];
		st[23] ^= bc[2];

		st[4] ^= bc[3];
		st[9] ^= bc[3];
		st[14] ^= bc[3];
		st[19] ^= bc[3];
		st[24] ^= bc[3];

		// Rho Pi
		t = st[1];
		#pragma unroll
		for (i = 0; i < 24; ++i) {
			bc[0] = st[keccakf_piln[i]];
			st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
			t = bc[0];
		}

		#pragma unroll
		for(int i = 0; i < 25; i += 5)
		{
			ulong tmp1 = st[i], tmp2 = st[i + 1];

			st[i] = bitselect(st[i] ^ st[i + 2], st[i], st[i + 1]);
			st[i + 1] = bitselect(st[i + 1] ^ st[i + 3], st[i + 1], st[i + 2]);
			st[i + 2] = bitselect(st[i + 2] ^ st[i + 4], st[i + 2], st[i + 3]);
			st[i + 3] = bitselect(st[i + 3] ^ tmp1, st[i + 3], st[i + 4]);
			st[i + 4] = bitselect(st[i + 4] ^ tmp2, st[i + 4], tmp1);
		}

		//  Iota
		st[0] ^= keccakf_rndc[round];
	}
}

void keccakf1600_2(__local ulong *st)
{
	int i, round;
	ulong t, bc[5];

	#pragma unroll 1
	for (round = 0; round < 24; ++round)
	{
		bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20] ^ rotate(st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], 1UL);
		bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21] ^ rotate(st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23], 1UL);
		bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22] ^ rotate(st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24], 1UL);
		bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23] ^ rotate(st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], 1UL);
		bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24] ^ rotate(st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21], 1UL);

		st[0] ^= bc[4];
		st[5] ^= bc[4];
		st[10] ^= bc[4];
		st[15] ^= bc[4];
		st[20] ^= bc[4];

		st[1] ^= bc[0];
		st[6] ^= bc[0];
		st[11] ^= bc[0];
		st[16] ^= bc[0];
		st[21] ^= bc[0];

		st[2] ^= bc[1];
		st[7] ^= bc[1];
		st[12] ^= bc[1];
		st[17] ^= bc[1];
		st[22] ^= bc[1];

		st[3] ^= bc[2];
		st[8] ^= bc[2];
		st[13] ^= bc[2];
		st[18] ^= bc[2];
		st[23] ^= bc[2];

		st[4] ^= bc[3];
		st[9] ^= bc[3];
		st[14] ^= bc[3];
		st[19] ^= bc[3];
		st[24] ^= bc[3];

		// Rho Pi
		t = st[1];
		#pragma unroll
		for (i = 0; i < 24; ++i) {
			bc[0] = st[keccakf_piln[i]];
			st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
			t = bc[0];
		}

		#pragma unroll
		for(int i = 0; i < 25; i += 5)
		{
			ulong tmp1 = st[i], tmp2 = st[i + 1];

			st[i] = bitselect(st[i] ^ st[i + 2], st[i], st[i + 1]);
			st[i + 1] = bitselect(st[i + 1] ^ st[i + 3], st[i + 1], st[i + 2]);
			st[i + 2] = bitselect(st[i + 2] ^ st[i + 4], st[i + 2], st[i + 3]);
			st[i + 3] = bitselect(st[i + 3] ^ tmp1, st[i + 3], st[i + 4]);
			st[i + 4] = bitselect(st[i + 4] ^ tmp2, st[i + 4], tmp1);
		}

		//  Iota
		st[0] ^= keccakf_rndc[round];
	}
}

#   define IDX(x)	(x)

inline uint getIdx()
{
	return get_global_id(0) - get_global_offset(0);
}

inline uint getIdy()
{
	return get_global_id(1) - get_global_offset(1);
}

inline float4 _mm_add_ps(float4 a, float4 b)
{
	return a + b;
}

inline float4 _mm_sub_ps(float4 a, float4 b)
{
	return a - b;
}

inline float4 _mm_mul_ps(float4 a, float4 b)
{

	//#pragma OPENCL SELECT_ROUNDING_MODE rte
	return a * b;
}

inline float4 _mm_div_ps(float4 a, float4 b)
{
	return a / b;
}

inline float4 _mm_and_ps(float4 a, int b)
{
	return as_float4(as_int4(a) & (int4)(b));
}

inline float4 _mm_or_ps(float4 a, int b)
{
	return as_float4(as_int4(a) | (int4)(b));
}

inline int4 _mm_alignr_epi8(int4 a, const uint rot)
{
	const uint right = 8 * rot;
	const uint left = (32 - 8 * rot);
	return (int4)(
		((uint)a.x >> right) | ( a.y << left ),
		((uint)a.y >> right) | ( a.z << left ),
		((uint)a.z >> right) | ( a.w << left ),
		((uint)a.w >> right) | ( a.x << left )
	);
}

inline global int4* scratchpad_ptr(uint idx, uint n, __global int *lpad) { return (__global int4*)((__global char*)lpad + (idx & MASK) + n * 16); }

inline float4 fma_break(float4 x)
{
	// Break the dependency chain by setitng the exp to ?????01
	x = _mm_and_ps(x, 0xFEFFFFFF);
	return _mm_or_ps(x, 0x00800000);
}

inline void sub_round(float4 n0, float4 n1, float4 n2, float4 n3, float4 rnd_c, float4* n, float4* d, float4* c)
{
	n1 = _mm_add_ps(n1, *c);
	float4 nn = _mm_mul_ps(n0, *c);
	nn = _mm_mul_ps(n1, _mm_mul_ps(nn,nn));
	nn = fma_break(nn);
	*n = _mm_add_ps(*n, nn);

	n3 = _mm_sub_ps(n3, *c);
	float4 dd = _mm_mul_ps(n2, *c);
	dd = _mm_mul_ps(n3, _mm_mul_ps(dd,dd));
	dd = fma_break(dd);
	*d = _mm_add_ps(*d, dd);

	//Constant feedback
	*c = _mm_add_ps(*c, rnd_c);
	*c = _mm_add_ps(*c, (float4)(0.734375f));
	float4 r = _mm_add_ps(nn, dd);
	r = _mm_and_ps(r, 0x807FFFFF);
	r = _mm_or_ps(r, 0x40000000);
	*c = _mm_add_ps(*c, r);

}

// 9*8 + 2 = 74
inline void round_compute(float4 n0, float4 n1, float4 n2, float4 n3, float4 rnd_c, float4* c, float4* r)
{
	float4 n = (float4)(0.0f);
	float4 d = (float4)(0.0f);

	sub_round(n0, n1, n2, n3, rnd_c, &n, &d, c);
	sub_round(n1, n2, n3, n0, rnd_c, &n, &d, c);
	sub_round(n2, n3, n0, n1, rnd_c, &n, &d, c);
	sub_round(n3, n0, n1, n2, rnd_c, &n, &d, c);
	sub_round(n3, n2, n1, n0, rnd_c, &n, &d, c);
	sub_round(n2, n1, n0, n3, rnd_c, &n, &d, c);
	sub_round(n1, n0, n3, n2, rnd_c, &n, &d, c);
	sub_round(n0, n3, n2, n1, rnd_c, &n, &d, c);

	// Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
	d = _mm_and_ps(d, 0xFF7FFFFF);
	d = _mm_or_ps(d, 0x40000000);
	*r =_mm_add_ps(*r, _mm_div_ps(n,d));
}

inline int4 single_comupte(float4 n0, float4 n1, float4 n2, float4 n3, float cnt, float4 rnd_c, __local float4* sum)
{
	float4 c= (float4)(cnt);
	// 35 maths calls follow (140 FLOPS)
	float4 r = (float4)(0.0f);

	for(int i = 0; i < 4; ++i)
		round_compute(n0, n1, n2, n3, rnd_c, &c, &r);

	// do a quick fmod by setting exp to 2
	r = _mm_and_ps(r, 0x807FFFFF);
	r = _mm_or_ps(r, 0x40000000);
	*sum = r; // 34
	float4 x = (float4)(536870880.0f);
	r = _mm_mul_ps(r, x); // 35
	return convert_int4_rte(r);
}

inline void single_comupte_wrap(const uint rot, int4 v0, int4 v1, int4 v2, int4 v3, float cnt, float4 rnd_c, __local float4* sum, __local int4* out)
{
	float4 n0 = convert_float4_rte(v0);
	float4 n1 = convert_float4_rte(v1);
	float4 n2 = convert_float4_rte(v2);
	float4 n3 = convert_float4_rte(v3);

	int4 r = single_comupte(n0, n1, n2, n3, cnt, rnd_c, sum);
	*out = rot == 0 ? r : _mm_alignr_epi8(r, rot);
}


static const __constant uint look[16][4] = {
	{0, 1, 2, 3},
	{0, 2, 3, 1},
	{0, 3, 1, 2},
	{0, 3, 2, 1},

	{1, 0, 2, 3},
	{1, 2, 3, 0},
	{1, 3, 0, 2},
	{1, 3, 2, 0},

	{2, 1, 0, 3},
	{2, 0, 3, 1},
	{2, 3, 1, 0},
	{2, 3, 0, 1},

	{3, 1, 2, 0},
	{3, 2, 0, 1},
	{3, 0, 1, 2},
	{3, 0, 2, 1}
};

static const __constant float ccnt[16] = {
	1.34375f,
	1.28125f,
	1.359375f,
	1.3671875f,

	1.4296875f,
	1.3984375f,
	1.3828125f,
	1.3046875f,

	1.4140625f,
	1.2734375f,
	1.2578125f,
	1.2890625f,

	1.3203125f,
	1.3515625f,
	1.3359375f,
	1.4609375f
};

struct SharedMemChunk
{
	int4 out[16];
	float4 va[16];
};

__attribute__((reqd_work_group_size(WORKSIZE * 16, 1, 1)))
__kernel void cn1_cn_gpu(__global int *lpad_in, __global int *spad, uint numThreads)
{
	const uint gIdx = getIdx();

#if(COMP_MODE==1)
	if(gIdx/16 >= numThreads)
		return;
#endif

	uint chunk = get_local_id(0) / 16;

	__global int* lpad = (__global int*)((__global char*)lpad_in + MEMORY * (gIdx/16));

	__local struct SharedMemChunk smem_in[WORKSIZE];
	__local struct SharedMemChunk* smem = smem_in + chunk;

	uint tid = get_local_id(0) % 16;

	uint idxHash = gIdx/16;
	uint s = ((__global uint*)spad)[idxHash * 50] >> 8;
	float4 vs = (float4)(0);

	// tid divided
	const uint tidd = tid / 4;
	// tid modulo
	const uint tidm = tid % 4;
	const uint block = tidd * 16 + tidm;

	#pragma unroll CN_UNROLL
	for(size_t i = 0; i < ITERATIONS; i++)
	{
		mem_fence(CLK_LOCAL_MEM_FENCE);
		int tmp = ((__global int*)scratchpad_ptr(s, tidd, lpad))[tidm];
		((__local int*)(smem->out))[tid] = tmp;
		mem_fence(CLK_LOCAL_MEM_FENCE);

		{
			single_comupte_wrap(
				tidm,
				*(smem->out + look[tid][0]),
				*(smem->out + look[tid][1]),
				*(smem->out + look[tid][2]),
				*(smem->out + look[tid][3]),
				ccnt[tid], vs, smem->va + tid,
				smem->out + tid
			);
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);

		int outXor = ((__local int*)smem->out)[block];
		for(uint dd = block + 4; dd < (tidd + 1) * 16; dd += 4)
			outXor ^= ((__local int*)smem->out)[dd];

		((__global int*)scratchpad_ptr(s, tidd, lpad))[tidm] = outXor ^ tmp;
		((__local int*)smem->out)[tid] = outXor;

		float va_tmp1 = ((__local float*)smem->va)[block] + ((__local float*)smem->va)[block + 4];
		float va_tmp2 = ((__local float*)smem->va)[block+ 8] + ((__local float*)smem->va)[block + 12];
		((__local float*)smem->va)[tid] = va_tmp1 + va_tmp2;

		mem_fence(CLK_LOCAL_MEM_FENCE);

		int out2 = ((__local int*)smem->out)[tid] ^ ((__local int*)smem->out)[tid + 4 ] ^ ((__local int*)smem->out)[tid + 8] ^ ((__local int*)smem->out)[tid + 12];
		va_tmp1 = ((__local float*)smem->va)[block] + ((__local float*)smem->va)[block + 4];
		va_tmp2 = ((__local float*)smem->va)[block + 8] + ((__local float*)smem->va)[block + 12];
		va_tmp1 = va_tmp1 + va_tmp2;
		va_tmp1 = fabs(va_tmp1);

		float xx = va_tmp1 * 16777216.0f;
		int xx_int = (int)xx;
		((__local int*)smem->out)[tid] = out2 ^ xx_int;
		((__local float*)smem->va)[tid] = va_tmp1 / 64.0f;

		mem_fence(CLK_LOCAL_MEM_FENCE);

		vs = smem->va[0];
		s = smem->out[0].x ^ smem->out[0].y ^ smem->out[0].z ^ smem->out[0].w;
	}
}


static const __constant uint skip[3] = {
	20,22,22
};

inline void generate_512(uint idx, __local ulong* in, __global ulong* out)
{
	ulong hash[25];

	hash[0] = in[0] ^ idx;
	for(int i = 1; i < 25; ++i)
		hash[i] = in[i];

	for(int a = 0; a < 3;++a)
	{
		keccakf1600_1(hash);
		for(int i = 0; i < skip[a]; ++i)
			out[i] = hash[i];
		out+=skip[a];
	}
}

// __attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void cn0_cn_gpu(__global ulong *input, __global int *Scratchpad, __global ulong *states, uint Threads, ulong extraNonce)
{
    const uint gIdx = getIdx();
    ulong State[25];

#if(COMP_MODE==1)
    // do not use early return here
	if(gIdx < Threads)
#endif
    {
        states += 25 * gIdx;

        Scratchpad = (__global int*)((__global char*)Scratchpad + MEMORY * gIdx);

// NVIDIA
#ifdef __NV_CL_C_VERSION
			for(uint i = 0; i < 8; ++i)
				State[i] = input[i];
#else
            ((ulong8 *)State)[0] = vload8(0, input);
#endif
            State[8]  = as_ulong(as_uchar8(extraNonce + get_global_id(0)).s76543210);
            // State[8]  = extraNonce + get_global_id(0);
            State[9]  = input[9];
            State[10] = input[10];

            for (int i = 11; i < 25; ++i) {
                State[i] = 0x00UL;
            }

            // Last bit of padding
            State[16] = 0x8000000000000000UL;

            keccakf1600_1(State);

            #pragma unroll
            for (int i = 0; i < 25; ++i) {
                states[i] = State[i];
            }
	}
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void cn00_cn_gpu(__global int *Scratchpad, __global ulong *states)
{
    const uint gIdx = getIdx() / 64;
    __local ulong State[25];

	states += 25 * gIdx;

    Scratchpad = (__global int*)((__global char*)Scratchpad + MEMORY * gIdx);

	for(int i = get_local_id(0); i < 25; i+=get_local_size(0))
		State[i] = states[i];

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint i = get_local_id(0); i < MEMORY / 512; i += get_local_size(0))
	{
		generate_512(i, State, (__global ulong*)((__global uchar*)Scratchpad + i*512));
	}
}




static const __constant uchar rcon[8] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40 };

#define SubWord(inw)		((sbox[BYTE(inw, 3)] << 24) | (sbox[BYTE(inw, 2)] << 16) | (sbox[BYTE(inw, 1)] << 8) | sbox[BYTE(inw, 0)])

void AESExpandKey256(uint *keybuf)
{
	//#pragma unroll 4
	for(uint c = 8, i = 1; c < 40; ++c)
	{
		// For 256-bit keys, an sbox permutation is done every other 4th uint generated, AND every 8th
		uint t = ((!(c & 7)) || ((c & 7) == 4)) ? SubWord(keybuf[c - 1]) : keybuf[c - 1];

		// If the uint we're generating has an index that is a multiple of 8, rotate and XOR with the round constant,
		// then XOR this with previously generated uint. If it's 4 after a multiple of 8, only the sbox permutation
		// is done, followed by the XOR. If neither are true, only the XOR with the previously generated uint is done.
		keybuf[c] = keybuf[c - 8] ^ ((!(c & 7)) ? rotate(t, 24U) ^ as_uint((uchar4)(rcon[i++], 0U, 0U, 0U)) : t);
	}
}

#if defined(__clang__)
#	if __has_builtin(__builtin_amdgcn_ds_bpermute)
#		define HAS_AMD_BPERMUTE  1
#	endif
#endif

__attribute__((reqd_work_group_size(8, WORKSIZE, 1)))
__kernel void cn2(__global uint4 *Scratchpad, __global ulong *states, __global uint *output, ulong Target, uint Threads)
{
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    uint ExpandedKey2[40];
    uint4 text;

    uint gIdx = get_global_id(1) - get_global_offset(1);
    uint groupIdx = get_local_id(1);
    uint lIdx = get_local_id(0);

    for (int i = groupIdx * 8 + lIdx; i < 256; i += get_local_size(0) * get_local_size(1)) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint4 xin1[WORKSIZE][8];
    __local uint4 xin2[WORKSIZE][8];

#if(COMP_MODE==1)
    // do not use early return here
    if(gIdx < Threads)
#endif
    {
        states += 25 * gIdx;
        Scratchpad += gIdx * (MEMORY >> 4);

        #if defined(__Tahiti__) || defined(__Pitcairn__)

        for(int i = 0; i < 4; ++i) ((ulong *)ExpandedKey2)[i] = states[i + 4];
        text = vload4(lIdx + 4, (__global uint *)states);

        #else
        text = vload4(lIdx + 4, (__global uint *)states);
        ((uint8 *)ExpandedKey2)[0] = vload8(1, (__global uint *)states);

        #endif

        AESExpandKey256(ExpandedKey2);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#	if (HAS_AMD_BPERMUTE == 1)
	int lane = (groupIdx * 8 + ((lIdx + 1) % 8)) << 2;
	uint4 tmp = (uint4)(0, 0, 0, 0);
#	else
    __local uint4* xin1_store = &xin1[groupIdx][lIdx];
    __local uint4* xin1_load = &xin1[groupIdx][(lIdx + 1) % 8];
    __local uint4* xin2_store = &xin2[groupIdx][lIdx];
    __local uint4* xin2_load = &xin2[groupIdx][(lIdx + 1) % 8];
    *xin2_store = (uint4)(0, 0, 0, 0);
#	endif

#if(COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#endif
    {

#	if	(HAS_AMD_BPERMUTE == 1)
        #pragma unroll 2
        for(int i = 0, i1 = lIdx; i < (MEMORY >> 7); ++i, i1 = (i1 + 16) % (MEMORY >> 4))
        {
            text ^= Scratchpad[IDX((uint)i1)];
			text ^= tmp;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            text.s0 ^= __builtin_amdgcn_ds_bpermute(lane, text.s0);
            text.s1 ^= __builtin_amdgcn_ds_bpermute(lane, text.s1);
            text.s2 ^= __builtin_amdgcn_ds_bpermute(lane, text.s2);
            text.s3 ^= __builtin_amdgcn_ds_bpermute(lane, text.s3);
			//__builtin_amdgcn_s_waitcnt(0);
            text ^= Scratchpad[IDX((uint)i1 + 8u)];

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
            tmp.s0 = __builtin_amdgcn_ds_bpermute(lane, text.s0);
            tmp.s1 = __builtin_amdgcn_ds_bpermute(lane, text.s1);
            tmp.s2 = __builtin_amdgcn_ds_bpermute(lane, text.s2);
            tmp.s3 = __builtin_amdgcn_ds_bpermute(lane, text.s3);
			//__builtin_amdgcn_s_waitcnt(0);
        }

        text ^= tmp;
#	else

		#pragma unroll 2
		for(int i = 0, i1 = lIdx; i < (MEMORY >> 7); ++i, i1 = (i1 + 16) % (MEMORY >> 4))
		{
			text ^= Scratchpad[IDX((uint)i1)];
			barrier(CLK_LOCAL_MEM_FENCE);
			text ^= *xin2_load;
			#pragma unroll 10
			for(int j = 0; j < 10; ++j)
			    text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
			*xin1_store = text;
			text ^= Scratchpad[IDX((uint)i1 + 8u)];
			barrier(CLK_LOCAL_MEM_FENCE);
			text ^= *xin1_load;

			#pragma unroll 10
			for(int j = 0; j < 10; ++j)
			    text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

			*xin2_store = text;
		}

        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin2_load;
#	endif

    }

    /* Also left over threads performe this loop.
     * The left over thread results will be ignored
     */
    #pragma unroll 16
    for(size_t i = 0; i < 16; i++)
    {
        #pragma unroll 10
        for (int j = 0; j < 10; ++j) {
            text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
        }
#if (HAS_AMD_BPERMUTE == 1)
	    text.s0 ^= __builtin_amdgcn_ds_bpermute(lane, text.s0);
        text.s1 ^= __builtin_amdgcn_ds_bpermute(lane, text.s1);
        text.s2 ^= __builtin_amdgcn_ds_bpermute(lane, text.s2);
        text.s3 ^= __builtin_amdgcn_ds_bpermute(lane, text.s3);
		//__builtin_amdgcn_s_waitcnt(0);
#else
        barrier(CLK_LOCAL_MEM_FENCE);
        *xin1_store = text;
        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin1_load;
#endif
    }

    __local ulong State_buf[8 * 25];
#if(COMP_MODE==1)
    // do not use early return here
    if(gIdx < Threads)
#endif
    {
        vstore2(as_ulong2(text), lIdx + 4, states);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

#if(COMP_MODE==1)
    // do not use early return here
    if(gIdx < Threads)
#endif
    {
        if(!lIdx)
        {
            __local ulong* State = State_buf + groupIdx * 25;

            for(int i = 0; i < 25; ++i) State[i] = states[i];

            keccakf1600_2(State);

			if(as_ulong(as_uchar8(State[0]).s76543210) <= Target)
			// if(State[3] <= Target)
			{
				ulong outIdx = atomic_inc(output + 0xFF);
				if(outIdx < 0xFF)
					output[outIdx] = getIdy();
			}
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}
