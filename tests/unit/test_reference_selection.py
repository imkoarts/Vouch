from app.services.draft_service import RetryDisposition, retry_disposition
from app.services.reference_selection import assess_reference_eligibility


def test_informative_position_and_market_observation_are_eligible() -> None:
    position = assess_reference_eligibility(
        "I sold the position after the average cost rose, but still believe in the ecosystem "
        "and see better uses for the capital right now."
    )
    observation = assess_reference_eligibility(
        "Markets do not require social validation to make money; the result is the validation."
    )

    assert position.eligible is True
    assert observation.eligible is True
    assert position.utility_score > 0.5
    assert observation.editorial_intent == "quote_reaction"


def test_user_screenshot_style_news_and_quant_posts_are_eligible() -> None:
    tariff_news = assess_reference_eligibility(
        "BREAKING: The U.S. announces it is holding Canada responsible for wildfire smoke "
        "entering the country and may add the costs to Canadian tariffs."
    )
    quant_result = assess_reference_eligibility(
        "A trader used Claude to build a quant bot and made $105,428 on Polymarket. It made "
        "31,240 predictions in 38 days with a 46% win rate and focused on short Up or Down markets."
    )
    debris = assess_reference_eligibility("@first @second haha nice tactic for the survivors")

    assert tariff_news.eligible and tariff_news.editorial_intent == "report_event"
    assert quant_result.eligible and quant_result.utility_score > 0.7
    assert debris.eligible is False


def test_mention_joke_url_only_and_low_context_reaction_are_rejected() -> None:
    samples = (
        "@one @two wow, the whale?",
        "https://t.co/synthetic",
        "@writer LMAO gov crime real",
        "happy to see a player getting credit too https://t.co/synthetic",
    )

    assert all(not assess_reference_eligibility(text).eligible for text in samples)


def test_low_information_gain_is_replanned_not_forced_into_no_post() -> None:
    disposition = retry_disposition(
        (
            "LOW_INFORMATION_GAIN",
            "PROMISED_INSIGHT_NOT_DELIVERED",
            "QUOTE_CONTEXT_REQUIRED",
            "VAGUE_USER_PROXY",
        )
    )

    assert disposition is RetryDisposition.REPLAN
