import pandas as pd

print
projects = {'bcel': ['a9c13ede0e565fae0593c1fde3b774d93abf3f71', 'bebe70de81f2f8912857ddb33e82d3ccc146a24e', 'bbaf623d750f030d186ed026dc201995145c63ec', 'fa271c5d7ed94dd1d9ef31c52f32d1746d5636dc', 'dce57b377d1ad6711ff613639303683e90f7bcc8', '9174edf0d530540c9f6df76b4d786c5a6ad78a5d', '3aecd517ad0ac4c83828a5f89b6b062acb6f4f6a', 'f38847e90714fbefc33042912d1282cc4fb7d43e', '893d9bbcdbd5ce764db1a38eccd73af150e5d34d', '9cd000cc265bfd0997e0277363dbe28ed8a28714', 'f70742fde892c2fac4f0862f8f0b00e121f7d16e', 'a303928530ee61e45b4523cd5894c9a8bdb9deaa', '647c723ba1262e1ffce520524692b366a7fde45a', 'fe98b6f098069607955a68f1d695031c011f6452', '5bfa4baa2b7b2cc3dc4cc2600bbcd5d74df7451c', '8fb97bd21c565e0da8300d6a87a95d0fe812bca8', 'daada0977098e6633de09fe4c73643ddd8331f06'],
'csv': ['8e25a2b30cae841101540c26ff21b79c51ad3eff', '660f7c9f853092ec8abf5d6c81d260e3c80c2194', 'c1c8b32809df295423fc897eae0e8b22bfadfe27', 'a227a1e2fb61ff5f192cfd8099e7e6f4848d7d43', '2596fdeebcab53fe459c481990bf1dec838128a5', 'f76a1357057cd3caaf9b0904d9cc57ce384658a3', '640b2f52dca971a977f146a32568ee00d33b45be', '03b98004527077686aea90b38a2bff39c6e9c998', '387d8109a352730e1795d74069cfd3745763abcf', '473e1b1eb9ceaa0fc956c0e76345fd1ab0575246', '0fbd1af5e3bd70454d5e398493a5c983aead2b67', 'b493818433b788238fe80fa15d9c6427e52b8590', 'f1d681e95a1c97449802e00895323366606067fb', 'a2875ade152087b733470d20075835b4f536d0bf', '0d44e840563a3a26a7b47c39789a48d23cc21634', '23d1fa4c7f29a757e5536e953071608ca4fd7869', 'b669d89f0cc86f0313859b35a2ec691279e09f58', '4ef853bc13d3548ce9cae02c7c26ddbc1c790669', '7916c61b0297b50fb6115ae0e8f6e6fc7b280966', '0ae402197cc67e0e103aaa1feb72a2e13faa5897', '222426dd79ed51fae3f3b372f12bdd35a6121e65'],
'gson': ['bb9a1f255aa7e7a07bf42f5bffe04f9204a59eb1', 'b6acf1178a1e9279a77235abe55d6895dd5c09f3', '6a368d89da37917be7714c3072b8378f4120110a', 'b41030d3dc965eca1b0a0f1c35ce185dd8e80e99'],
'openfire': ['c0f2614b1bcfbdcad337128978aa9ae82e66fc50', 'd7dfd04eca85ca8933a408c7bc99eb6ed8fb8599', 'e020f58943742b0be541bafd30808d587867712b', 'f5aece5bbdd81bbab824e3995cbdfd96920766f2', '5b6b732fda44fbc3c38a7036b1ed7522d9f8b129', '348185c8385cf8923455a71dd614d7ee09ee7bf0', '1bcb80f9f23d70fbd7ee8a833abe02b2305f848e', 'cb90fc79edb71b888d3ee25f2f59da3be2e72174', 'c4c568e13715a3cdb980e340352ec4fb30c61fae', '2bc37e6f1d48c84228efe2ccd360148cc9be46f0', '9b35e20028cdb52e29e4ac815a8419ffb35ca2d3', 'b61bce39e3a0ce786e4464706ea2e9b4c5be7a77', '18a7bb6262b4237f9c80071d24234f5d9b2e08fc', '13e73ed3d7fc00646c642d02b66b50a86cd1c42b', '18257f671cef3c8340be950a4d5c50ef3b31d9f6', '92a9f2e9e2c78aabdf446af0c680c7c0991f807a', '41041642078a1e498c4f8b6e6a5b226409b5dbcd', '3acceb064aeb3e6a2b3306ec35ceebc1e3cc8c8f', '586a86b6c0ac0121dff8edd69e0ab79b0f07d2ca', '5b55500c24f27f2296dc80cba2765f13cdf6d5d3', '4235b9eebdc5ae28713707106c97b922145d0e67', '0db7fbcab383272658507241df9aaccfce63a20a', '8d975c8f006a69451ec4d4fbd951315b44f50b88', '2519596205da62daff32ab25188fead966cb2653', '5e5d9e58eb05764f50e5d2b03ee7416dab9bb6a1', 'cd0a57352084fceee70818b920dbe1ed4e331b62', '2eeed5a8d6d90e1c14a18ff697d94125e5cb4da4', '9baa41983346187d9abcc9dfb0f45f2285a84bc4', 'a770995b151fb92509ece3e040ce929f63d27969', '41b1a6cc1852eb2648de99afcafa1d95678d5208', '38480e6376a65983d9057b17b7063b4c9c68aff0', '0bde7625668738bdee0e48f5fcb821ed91a71ec2', '64f46292cf130280b5ca7a3ad54bc2fecec01fc6', '30b8ad764a15c5e9755a924b2f16ef3bb48d8b69', 'dcef698f85eb09c46a7ca3865d6f105610909ba5', '244f6cfc4cbf34fc9beb09567cdc675c7e97c41c', '0390a6b8fc0a81f7923d5fbfc78a765d5cb97436', 'daf5ef478340c0f9d88dd43768d11f119f88c608', '1b59dd22ec05b87636cd7422424f8be823cb55ba', '8ceafc3b0071d816cc640b3a2a16793e34f4a2ac', 'c689f1576d210f5c288aebd860d9c0531fe6a78d', '9e5bfda5e7316855e1206ad004cc4135a08a1563', '2a6303d8695b1375f921029fbc233b72eafd64a2', '2d7a8d0e747e9630abcf27b154d0732d13b06269', '538cb6113be741c2178e24ec489664419480ceff', 'ed7eb5102f89d589d39d227e09da09074176aa6e', '8c7157691883694df4f1479f34472f7b405a16f4', '010973b082e8b504241c22ede1dc7073cc33ac19', '113917ac5f1750b216dae9f8d21e998dd8ceb39d', '7c5b4d791574c3b83d957f8a4380e8ef3e0482ae', '46004515dffd63835fdcee87ebac76abde4afef6', '8b8c2c6402e396d36a552f91eed6bfd04eba9b87', '7934f499bc7d03b1ef2104f666b73bfb7d733602', '77ba63948844efdea31d9d2d024d81f8fdcc38d6', '6e4a0edb62d73db6350c25a19e157eb8b9ecc6a5', 'a31a6cee02bfa93fdf055bcb0f97be157bb342ca', 'fc144a21e53f7ad10232394a51784f77d4d7e9a4', 'e372d3688731d9d0c4e38b951e319ad189a91376', 'bd14e188131ca0bce12515c2aaa04b444710613f', '1d6aaa125ffa4eb65d4ce2baaf49a7c51eef932d', 'ba56ca5cea9d025c0d9f4e01c2cce046002eed9b', '32098091265f56407d4061cd50847d6cf170936d', '3acbbf9a9885e81326ba83db26473608f2adc094', '6c1d6b387048fc2b12dda2c7f32ee9fa4b87dd2e', '212092be5b09041217673123a3946e753da9fda7', 'ba59155108efc86c8a2fb28eac254001c3487df6', '7142df7c85e6f18dc8306cf8cb973790feddecb5', '45484399372699ff7775f5e3335a74d7fe4763df', '1cd0601df4795dfb8d076667f1b1586ef9250f2f', '94b2cc3d3bef562d36e62e16a10680ccccd7c0bb', 'ca69445887fd8b4cbcc693a655e7e9d0a84406b6', '9e35a6e69cc9fcebaa8e0362829d6068713ea061', '02433aed14d04a18b8cb99ce8e6a2ad3c41e5e97', 'c77641903f599a8d0841f603914d3d70a3320774', '90330ddba1fdaeae847cf80a4996ea005ab03fea', '58b6253216de23ef1fea87606bcd91c044d4a83b', 'b7b6a2f18d595cb55c0018f7881448231a70208d', 'cbe17f99fb5b106f7a2dc5c1fea96b47bcdfe9d1', '1ea6ab11e1aaa441ea47d890e994e2fc612c5342', 'dfa1cb85bfb0988874a0861ab7df4eaf40a0c772', '0ded93971435cd4352cb19b3e50c01d055c88754', '29a9f3d8ba6bb22aeeb3384a94eb2cca024ad157', '598bcf74a44a4246adc57018fb46922a41e155bd', 'e7078a55626cc60666d8f6bcc7b26660503dc2dc'],
'text': ['cb85bed468e99d34b88d0c81fe20eb3b1615660e', 'b63df8d66e8306b2608c16be3661248348e78a2f', '3866d2626dc3767b003c7cfe163a388b10c80957', '7643b12421100d29fd2b78053e77bcb04a251b2e', '4736b16d0e644289f3106275ebb1315750234e40', 'e1af89b53855f2f19138cbc3e8a49ca179c3d8f8', 'd3d93c4e68ce5d8c25aecbfff9d17017594bf3f2', 'ba44287bdd17a709523364820495760645da85b9', 'd9229655fdb335a9730bef6df789a46faa764b8a', '4ac48357180c9222f032dd8b055d90e1192e4a47', '44a9f06abfbe898361bd2b486f786b22329face6', '206931af2ad7084044aba19591a5034b634c23b1', 'e38039a3da2244741f5d33ab1b05bdee51c53c3e', '65e4314fbd6c3a8f5c248d07a4ccffc1f0ea8bb9', '230e868e6e33d02a162c4214cb38e117c9584473', '61eb9d01d3be74f692f9f9be5a988bdcbc09f6ec', '3a9fe391b4cbca59c7abb6aec1035ee87d1cf2f9'],
'io': ['8985de8fe74f6622a419b37a6eed0dbc484dc128', 'a73895fbefd57c23595a5e9e85f0649993c59080', '75f20dca72656225d0dc8e7c982e40caa9277d42', 'fa59009aaabcf8671a8d741993ef355f42b95ccd', '2e6f1e306edd81f3d0c7d5d36f4e1de84710bd08', '6efbccc88318d15c0f5fdcfa0b87e3dc980dca22', '2ae025fe5c4a7d2046c53072b0898e37a079fe62', 'a219081780bb1714876ef3e1109283b96f3b007b', '79e236e0508c43519a7ff7868e8d48883628b6e4', 'c5ef3334e968c2492a10a00bbba4186ba864c37c', '4077158829de92987367d3149e4ba71356bb5390', 'f49c21bdf2d48ca20b0f125b10f6d39326e2bada', 'd2d9fd18134e9724f7bea0cc63d10160ca739975', '8c83ba1bae3d97f10c8472b19fe26061e7843a09', 'cbb192dceba30e10ae8e8fafa98811e0014e2734', 'e36d53170875d26d59ca94bd376bf40bc5690ee6'],
'pdfbox': ['af337d52d731ea78910ed0351d0a07eeadbb4146', '10d1e91af4eb9a06af7e95460533bf3ebc1b1280', 'f28f2f19a3c6c73448c344d6909cdc0e9572d5ed', '42de6a08b5127b6364d5d572733819a84b9ec07d', '8876e8e1a0adbf619cef4638cc3cea073e3ca484', '8f23f8791c3a526c22cf6a0f6f0d19d830ced0d0', '9722f3dc57a24e421ce1f3c1fbd0298fc43415ad', '8bea7c87bdcc7f4ffbf56d911b6e38bebe5972c0', 'fa8029d80e39ff4febd1ce0d6ecc628bf2970ca0', '67e8845cd3fcba3be36896a2552b492a1749d349', 'f0dc0793c79293f548e99df140ed7352303c9472', '728ff429df7f75cdf142ae309edc58c15f655c8a', 'b32eae9ad0def46ddeba093e474fadd82e5503f3', 'cecb162611ad5c72cbf6b38ee8c05b909fdc12bc', '28468427911d9535c712a1c3149208a58dfc0d34', '983a4af646dd88a99f166632b9acb82b65cf1f4b', 'ee4cf8e7ac7e0be02293bb1df28a65d7ad73c31d', 'adcced4ba4496cce33fee729226f71c5c41a7fbf', '3a7c86d4c6abfe10f62e38b175c70d8fd0809ce5', '0cedd70a1bfc07e0e2bfbcdccf53f01cef27b0de', '2d374c65b206469c725b790a9783d389e57d2b7e', '618f05e1f4650b4a856cb2341b3c8dc99c363eb9', '7a997954b5f67d30aa00371bc6465d33744fa3eb', '402d15ba73a4b394efebe3b8703893e0c2f5b409', 'ba299e77314967a5e5c806bac6b918d95c2147f0', '4983455ee89c1378aa83ff5b49057960f965f5b8', '58c362bee6662e251bde0da9b8a33e8f247ba0f4', 'bc2f3322eaf7ea462f8678939ee60e31c656161e', '9b2e8e73b853d38490de98041627a3f9b075eb96', '1400def5c88140cc9be1245b9b4774a9d558d73c'],
'dubbo': ['db4007e44527451ceda23aa109b4123949b4210e', 'e4abc7a4ee67a8cbd184aef442d82518e67d84ae', 'd8f9768fe5f8192d690e5d09a40d19ed062932d5', 'e7894ca374e966a1d807e34b2744f276b843f39f', '2323dc8f8c718f52d2dd3fa2075d1b7497cc65e0', 'b4ba39f728bd34f7539adf9aca2fd08910b13e68', 'e915918e5c43319e1a7e2cab547685658d98b36a', '7de8b982fbd9e080edb91bf6ffd774ab519c0226', 'fff0047b1deca36042f387832d235717070b577a', '2759f386b1c91f284d2afbb478b6a1943885ce65', '767620aeddfc84d2299e8ab6f754e56a5ce5a265', '0aee8101665262c64122e5fa2c683ef834faff10', '5a5e3f3fd7d0482b6124bd5b52b10f4900856438', 'ed39e491c32b82591d09972299621570071ca8f3', '00ca43643d243f0866389abd294c6160e155c8a5', '42c3a07db376295b371653942ca316435e787434', 'c8c30d13f99c5270278b8fdeb8c04af4e08bc7b0', '6e61380e487550f7800e0e3df7649517af768dd6', '9c49efeacfd87d2d4409fb000cebd58e1114ec8a', 'aa3fe7b74dfe282ab7c3b83b2fc5559d06c8f9b4', '236a45694e9a1925e0a57f42686996f6fbf4c694', 'e8eeddbf42952e7dd4ba4428548f91006900ea5c', 'd38805030c2822abb8c3f04490074bfde874910b', '30616ea7fa945da7d76b703c2c2d29d123a6dfb0', 'de9fc426cf57a74fab0d0cd37fabb0d9cf6086bd', '95b776385ee8dcaca91568c11765e73d9b29b861', 'f0483b80a32e813a4393879744e30d0084c3eafd', '0be2a1bbbf9168490acecaf1eed1bd16cb8db402', 'd895bf15d0dda6a69c552c6b06d5e452692bef53', '96de4d9edaa15432e1f67b2d62dab8da9bfb96a1', 'af4cff2846bdfb9f105bf32b988616a4ed40a072', '5a6a069986f3627f2d50720c50024df31a3dc38a', '5ae160ad7b6b1ec67dad14bce819e5d27353cf21', '1d7e8fac4523d7482d5f2d8f25dd024807e3d3d5', 'c3e122e00e9607060dd10531108e563f16ac4f7f', 'eac84b290f82bb5346a449a973315b6860a9e665', '501e4dbdf9d93b8d5b59400cff3bdac5ea89fbc4', 'a36cc7520e0150d07d30e8baa0c61d5f9f11e6ed', 'ba7f6f38c36675268ba64f21e97d02bda7a731dc', '614bcebc01336ee5047a98f96b28915680c0399c', '38e0f15be33a5edb35b45d2c5d7c6be753bdd888', '88037747a3b69d3225c73f6fbcda36ebd8435887', '6ce7b11f982c5f7beb2ac5897fb33d48da357ada', '5eeb240337ccfbc820d4bde023d8cf643f33d735', '5fb3dbdc141c7bd85916221031b89a5d7be70fb8', '04576ff4ffdd5872e738b4e10e3905fe513109d2', '4f3017c71849a31de6a9f87e8e71a4d7dd084582', '22bb9cbcf75cab6445b8706574a986517855b535', '55a68a337b35a5940122b9afc774b47c52787214', 'e2d63ad98a278e05ab13ee75dbff9ef302709cf2', '99331ff50bfa478a343b4c23976886fb4710273f', '3deca192444b2667284f41a259fc6bcdd7a049d0', '0735bcb4ab38be0534b6f8fefd27c0cec9002d27', '0e1752e34e22c76dd57935c883cecbd8a8f3709c']
}

print('starting to check the commits...')


def check_commits(file):
    ck_df = pd.read_csv(file)
    print('### ' + file + '###')
    for project_name, versions in projects.items():
        print('--- ' + project_name + ' ---')
        for ver in versions:
            if not ver in ck_df['commit_hash'].unique():
                print(project_name, ver)
    print('##########')



ck = 'results/ck_all.csv'
check_commits(ck)

ck = 'results/und_all.csv'
check_commits(ck)
ck = 'results/changedistiller_all.csv'
check_commits(ck)
ck = 'results/organic_all.csv'
check_commits(ck)
ck = 'results/evometrics_all.csv'
check_commits(ck)
ck = 'results/refactoring_all.csv'

print('end')