// SPDX-FileCopyrightText: © 2026 Benedikt Nimmervoll
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use fotema_core::FaceId;
use fotema_core::database;
use fotema_core::people;
fn setup() -> people::Repository {
    let con = database::setup_in_memory().unwrap();
    let con = Arc::new(Mutex::new(con));

    let con_guard = con.lock().unwrap();
    con_guard
        .execute(
            "INSERT INTO pictures (
                picture_id, picture_path_b64, picture_path_lossy, link_path_b64,
                link_path_lossy, fs_created_ts
             ) VALUES (1, 'cGljLmpwZw==', 'pic.jpg', 'bGluaw==', 'link', '2026-01-01 00:00:00')",
            [],
        )
        .unwrap();
    con_guard
        .execute(
            "INSERT INTO people (person_id, name)
             VALUES (1, 'Anna')",
            [],
        )
        .unwrap();
    con_guard
        .execute(
            "INSERT INTO pictures_faces (
                face_id, model_name, picture_id, person_id, is_confirmed,
                thumbnail_path, bounds_path,
                bounds_x, bounds_y, bounds_width, bounds_height,
                right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                nose_x, nose_y, right_mouth_corner_x, right_mouth_corner_y,
                left_mouth_corner_x, left_mouth_corner_y, confidence
             ) VALUES (
                1, 'test', 1, 1, 1,
                'face.jpg', 'bounds.jpg',
                0.1, 0.1, 0.2, 0.2,
                0.1, 0.1, 0.2, 0.1,
                0.15, 0.15, 0.12, 0.18, 0.18, 0.18, 0.99
             )",
            [],
        )
        .unwrap();
    drop(con_guard);

    people::Repository::open(
        std::path::Path::new("/tmp"),
        std::path::Path::new("/tmp"),
        con,
    )
    .unwrap()
}

#[test]
fn a_rejected_face_is_remembered_and_detached() {
    let mut repo = setup();

    repo.mark_not_person(FaceId::new(1)).unwrap();

    let negatives = repo.find_negative_associations().unwrap();
    assert!(negatives.get(&1).is_some_and(|set| set.contains(&1)));

    // The face is back in the unknown pool.
    let unnamed = repo.find_unnamed_faces().unwrap();
    assert!(unnamed.iter().any(|face| face.face_id == FaceId::new(1)));
}
