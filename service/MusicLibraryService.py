import hashlib
from typing import List, Optional
from domain.User import User
from domain.Track import Track
from repository.RepositoryException import RepositoryException


class MusicLibraryService:
    def __init__(self, user_repo, track_repo):
        self.user_repo = user_repo
        self.track_repo = track_repo

    # --- USER ACTIONS ---

    def _hash_password(self, password: str) -> str:
        """Private helper to ensure we never handle plain-text passwords."""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, email, password) -> User:
        """Hashes password and creates a new user."""
        hashed_pw = self._hash_password(password)
        new_user = User(username=username, email=email, password=hashed_pw)
        return self.user_repo.create(new_user)

    def login(self, username, password) -> Optional[User]:
        """Verifies credentials and returns the User object or None."""
        hashed_pw = self._hash_password(password)
        return self.user_repo.find_user(username, hashed_pw)

    # --- TRACK & CLASSIFICATION ACTIONS ---

    def add_track(self, user_id: int, title: str, main_genre: str, sub_genre: str, features: List[float]) -> Track:

        # Step 3: Create Track Domain Object
        new_track = Track(
            user_id=user_id,
            title=title,
            main_genre=main_genre,
            sub_genre=sub_genre,
            features=features
        )

        # Step 4: Save to Database
        return self.track_repo.add(new_track)

    def update_track_info(self, track: Track):
        """Updates metadata for an existing track."""
        return self.track_repo.modify(track)

    def remove_track(self, user_id: int, track_id: int):
        """Deletes a track from history."""
        return self.track_repo.delete(user_id, track_id)

    # --- FIND ACTIONS (User-Scoped) ---

    def get_user_library(self, user_id: int) -> List[Track]:
        """Returns all tracks for a specific user."""
        return self.track_repo.find_all_by_user(user_id)

    def search_by_genre(self, user_id: int, genre: str) -> List[Track]:
        """Finds tracks of a specific genre within a user's collection."""
        return self.track_repo.find_by_main_genre(user_id, genre)

    def search_by_title(self, user_id: int, query: str) -> List[Track]:
        """Finds tracks by name within a user's collection."""
        return self.track_repo.find_by_title(user_id, query)