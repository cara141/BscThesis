import os

import pytest
import sqlite3
from repository.UserRepository import UserRepository
from repository.TrackRepository import TrackRepository
from repository.RepositoryException import RepositoryException
from domain.User import User
from domain.Track import Track


@pytest.fixture(scope="function")
def test_db():
    db_file = "test_mgc.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    yield db_file


@pytest.fixture
def repos(test_db):
    u_repo = UserRepository(test_db)
    t_repo = TrackRepository(test_db)
    return u_repo, t_repo


def test_database_schema_initialization(test_db):
    """Verifies that tables are actually created on init."""
    UserRepository(test_db)
    TrackRepository(test_db)

    with sqlite3.connect(test_db) as con:
        cursor = con.cursor()
        # query sqlite_master to see if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        assert "users" in tables
        assert "tracks" in tables


def test_full_user_track_integration(repos):
    u_repo, t_repo = repos

    u = User()
    u.username = "tester"
    u.email = "test@test.com"
    u.password = "hash123"
    saved_user = u_repo.create(u)

    t = Track(
        user_id=saved_user.id,
        title="Integration Song",
        main_genre="Electronic",
        sub_genre="Techno",
        features=[1.0] * 518
    )
    saved_track = t_repo.add(t)

    assert saved_track.id is not None
    assert saved_track.user_id == saved_user.id


def test_cascading_delete(repos):
    """
    Tests that deleting a user also deletes their tracks
    """
    u_repo, t_repo = repos

    u = u_repo.create(User(username="bye", email="b@b.com", password="p"))
    t_repo.add(Track(user_id=u.id, title="Gone", features=[]))

    assert len(t_repo.find_all_by_user(u.id)) == 1

    with sqlite3.connect(u_repo.db_path) as con:
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("DELETE FROM users WHERE id = ?", (u.id,))
        con.commit()

    assert len(t_repo.find_all_by_user(u.id)) == 0