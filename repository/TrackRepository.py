import sqlite3
import json
from domain.Track import Track
from repository.RepositoryException import RepositoryException


def _row_to_track(row) -> Track:
    return Track(
        id=row[0],
        user_id=row[1],
        title=row[2],
        main_genre=row[3],
        sub_genre=row[4],
        features=json.loads(row[5]) if row[5] else []
    )


class TrackRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute('''PRAGMA foreign_keys=ON''')

                cursor = con.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tracks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        main_genre TEXT,
                        sub_genre TEXT,
                        features TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                            ON DELETE CASCADE
                    )
                ''')
                con.commit()
                print("Tracks table has been created")
        except sqlite3.Error as e:
            raise RepositoryException(f"Database initialization failed: {str(e)}")

    def add(self, track: Track) -> Track:
        """Add a track entry to the database"""
        # we assume user_id is already set in the track object by the Service layer
        try:
            with sqlite3.connect(self.db_path) as con:
                cursor = con.cursor()
                cursor.execute('''
                    INSERT INTO tracks (user_id, title, main_genre, sub_genre, features)
                    VALUES (?, ?, ?, ?, ?)
                ''', (track.user_id, track.title, track.main_genre, track.sub_genre, json.dumps(track.features)))
                con.commit()
                track.id = cursor.lastrowid
                return track
        except sqlite3.Error as e:
            raise RepositoryException(f"Error adding track: {e}")

    def find_all_by_user(self, user_id: str) -> list[Track]:
        """Fetch all tracks for a specific user."""
        with sqlite3.connect(self.db_path) as con:
            cursor = con.cursor()
            cursor.execute("SELECT * FROM tracks WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()
            return [_row_to_track(row) for row in rows]

    def find_by_main_genre(self, user_id: str, main_genre: str) -> list[Track]:
        """Search tracks by genre, but only for the specific user."""
        with sqlite3.connect(self.db_path) as con:
            cursor = con.cursor()
            cursor.execute('''
                SELECT * FROM tracks 
                WHERE user_id = ? AND main_genre = ?
            ''', (user_id, main_genre))
            rows = cursor.fetchall()
            return [_row_to_track(row) for row in rows]

    def find_by_title(self, user_id: str, title: str) -> list[Track]:
        """Search tracks by title, but only for the specific user."""
        with sqlite3.connect(self.db_path) as con:
            cursor = con.cursor()
            cursor.execute('''
                SELECT * FROM tracks 
                WHERE user_id = ? AND title LIKE ?
            ''', (user_id, f"%{title}%"))
            rows = cursor.fetchall()
            return [_row_to_track(row) for row in rows]

    def delete(self, user_id: str, track_id: int):
        """Delete a track, ensuring the user owns it."""
        with sqlite3.connect(self.db_path) as con:
            cursor = con.cursor()
            cursor.execute("DELETE FROM tracks WHERE user_id = ? AND id = ?", (user_id, track_id))
            con.commit()

    def modify(self, track: Track):
        """
        Updates an existing track record in the database.
        Requires the track.id to be populated.
        """
        if not track.id:
            raise RepositoryException("Cannot modify a track without a valid ID.")

        try:
            with sqlite3.connect(self.db_path) as con:
                # ensure foreign keys are enabled if you are changing user_id
                con.execute("PRAGMA foreign_keys = ON;")
                cursor = con.cursor()

                cursor.execute('''
                               UPDATE tracks
                               SET user_id    = ?,
                                   title      = ?,
                                   main_genre = ?,
                                   sub_genre  = ?,
                                   features   = ?
                               WHERE id = ?
                               ''', (
                                   track.user_id,
                                   track.title,
                                   track.main_genre,
                                   track.sub_genre,
                                   json.dumps(track.features),
                                   track.id
                               ))

                if cursor.rowcount == 0:
                    raise RepositoryException(f"No track found with ID {track.id}")

                con.commit()
        except sqlite3.Error as e:
            raise RepositoryException(f"Database error during update: {e}")