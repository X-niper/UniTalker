dataset_config = {
    'D0': {
        'dirname': 'BIWI',
        'annot_type': 'BIWI_23370_vertices',
        'scale': 0.2,
        'annot_dim': 23370 * 3,
        'subjects': 6,
        'pca': True,
    },
    'D1': {
        'dirname': 'vocaset',
        'annot_type': 'FLAME_5023_vertices',
        'scale': 1.0,
        'annot_dim': 5023 * 3,
        'subjects': 12,
        'pca': True,
    },
    'D2': {
        'dirname': 'meshtalk',
        'annot_type': 'meshtalk_6172_vertices',
        'scale': 0.001,
        'annot_dim': 6172 * 3,
        'subjects': 13,
        'pca': True,
    },
    'D3': {
        'dirname': '3DETF/HDTF',
        'annot_type': '3DETF_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 52,
        'subjects': 141,
        'pca': False,
    },
    'D4': {
        'dirname': '3DETF/RAVDESS',
        'annot_type': '3DETF_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 52,
        'subjects': 24,
        'pca': False,
    },
    'D5': {
        'dirname': 'D5_unitalker_faceforensics++',
        'annot_type': 'flame_params_from_dadhead',
        'scale': 1.0,
        'annot_dim': 413,
        'subjects': 719,
        'pca': False,
    },
    'D6': {
        'dirname': 'D6_unitalker_Chinese_speech',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 8,
        'pca': False,
    },
    'D7': {
        'dirname': 'D7_unitalker_song',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 11,
        'pca': False,
    },
}
