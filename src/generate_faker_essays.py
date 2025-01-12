import random

from faker import Faker

NB_WORDS = 70
VARIABLE_NB_WORDS = False
NB_ESSAYS = 2000
SEED = 4444

random.seed(SEED)
Faker.seed(SEED)
fake = Faker(locale="en")


def generate_faker_essays(
    nb_essays: int = 2000,
    nb_words: int = 70,
    variable_nb_words=False,
) -> list[str]:
    fake_essays = set()
    while len(fake_essays) < nb_essays:
        essay = fake.sentence(
            nb_words=nb_words,
            variable_nb_words=variable_nb_words,
        )
        if len(essay) <= 450:
            fake_essays.add(essay)

    return list(fake_essays)


if __name__ == "__main__":
    file_id = f"{NB_ESSAYS}_{NB_WORDS}_{SEED}"

    fake_essays = generate_faker_essays(
        nb_essays=NB_ESSAYS,
        nb_words=NB_WORDS,
        variable_nb_words=VARIABLE_NB_WORDS,
    )

    with open(
        f"./data/faker_essays_{file_id}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        for essay in fake_essays:
            file.write(essay + "\n")
