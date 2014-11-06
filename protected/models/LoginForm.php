<?php

/**
 * LoginForm class.
 * LoginForm is the data structure for keeping
 * user login form data. It is used by the 'login' action of 'SiteController'.
 */
class LoginForm extends CFormModel
{
	public $Username;
	public $password;
	public $rememberMe;
        private $_identity;

	/**
	 * Declares the validation rules.
	 * The rules state that Username and password are required,
	 * and password needs to be authenticated.
	 */
	public function rules()
	{
		return array(
			// Username and password are required
			array('Username, password', 'required'),
			// rememberMe needs to be a boolean
			array('rememberMe', 'boolean'),
			// password needs to be authenticated
			array('password', 'authenticate'),
		);
	}

	/**
	 * Declares attribute labels.
	 */
	public function attributeLabels()
	{
		return array(
			'rememberMe'=>'Mantener la sesión iniciada',
		);
	}

	/**
	 * Authenticates the password.
	 * This is the 'authenticate' validator as declared in rules().
	 */
	public function authenticate($attribute,$params)
	{
		if(!$this->hasErrors()){
			$this->_identity=new UserIdentity($this->Username,$this->password);
			if(!$this->_identity->authenticate())
				$this->addError('password','Password incorrecto.');
                            //$identity->authenticate();
            switch ($this->_identity->errorCode) {
                case UserIdentity::ERROR_NONE:
                    $duration = $this->rememberMe ? 3600 * 24 * 360 : 0; // 360 days
                    Yii::app()->user->login($this->_identity, $duration);
                    break;
                case UserIdentity::ERROR_USERNAME_INVALID:
                    $this->addError('Username', 'Username incorrecto.');
                    break;
                default: // UserIdentity::ERROR_PASSWORD_INVALID
                 case UserIdentity::ERROR_PASSWORD_INVALID:
                    $this->addError('password', 'Password incorrecto.');
                    break;
                default: // UserIdentity::ERROR_PASSWORD_INVALID
                    $this->addError('password', 'Password incorrecto.');
                    break;
            }
		}
	}

	/**
	 * Logs in the user using the given Username and password in the model.
	 * @return boolean whether login is successful
	 */
	public function login()
	{
		if($this->_identity===null)
		{
			$this->_identity=new UserIdentity($this->Username,$this->password);
			$this->_identity->authenticate();
		}
		if($this->_identity->errorCode===UserIdentity::ERROR_NONE)
		{
			$duration=$this->rememberMe ? 3600*24*30 : 0; // 30 days
			Yii::app()->user->login($this->_identity,$duration);
			return true;
		}
		else
			return false;
	}
}
