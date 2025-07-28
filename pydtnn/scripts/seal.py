from tenseal import sealapi

slots = 4096
poly_degree = slots * 2
scale = 2 ** 40
level = sealapi.SEC_LEVEL_TYPE.TC128

modulus = sealapi.CoeffModulus.BFVDefault(poly_degree, level)

parameters = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.CKKS)
parameters.set_coeff_modulus(modulus)
parameters.set_poly_modulus_degree(poly_degree)

context = sealapi.SEALContext(parameters, True, level)

keygen = sealapi.KeyGenerator(context)

secret = keygen.secret_key()
public = sealapi.PublicKey()
keygen.create_public_key(public)
gelios = keygen.create_galois_keys()
relin = keygen.create_relin_keys()

encoder = sealapi.CKKSEncoder(context)
encryptor = sealapi.Encryptor(context, public, secret)
decryptor = sealapi.Decryptor(context, secret)

plain1 = sealapi.Plaintext()
cipher1 = sealapi.Ciphertext()
encoder.encode([1.0, 2.0, 3.0], scale, plain1)
encryptor.encrypt(plain1, cipher1)

plain2 = sealapi.Plaintext()
cipher2 = sealapi.Ciphertext()
encoder.encode([4.0, 5.0, 6.0], scale, plain2)
encryptor.encrypt(plain2, cipher2)

evaluator = sealapi.Evaluator(context)

cipher3 = sealapi.Ciphertext()
evaluator.add(cipher1, cipher2, cipher3)

plain3 = sealapi.Plaintext()
decryptor.decrypt(cipher3, plain3)

val = encoder.decode_double(plain3)[:3]

print(val)
