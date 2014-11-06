<?php

class Usuarios extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'usuarios';
	}

	public function rules()
	{
		return array(
			array('Username, Password, Analista', 'required'),
			array('Username, Password, Analista, Proceso', 'length', 'max'=>128),
                        array('Es_administrador,Es_analista','numerical', 'integerOnly'=>true),
			array('id, Username, Password, Analista, Proceso,Es_administrador,Es_analista', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'anotaciones' => array(self::HAS_MANY, 'Anotaciones', 'usuario'),
			'autorizaciones' => array(self::HAS_MANY, 'Autorizaciones', 'usuario'),
			'competenciasUsuarioses' => array(self::HAS_MANY, 'CompetenciasUsuarios', 'usuario'),
			'evaluaciones' => array(self::HAS_MANY, 'Evaluaciones', 'usuario'),
			'eventoses' => array(self::HAS_MANY, 'Eventos', 'usuario'),
			'metaDocs' => array(self::HAS_MANY, 'MetaDocs', 'usuario'),
			'metaDocs1' => array(self::HAS_MANY, 'MetaDocs', 'userRevisado'),
			'permisoses' => array(self::HAS_MANY, 'Permisos', 'usuario'),
			'prestamoses' => array(self::HAS_MANY, 'Prestamos', 'usuario'),
			'prestamoses1' => array(self::HAS_MANY, 'Prestamos', 'usuarioRcv'),
			'ubicacionTecs' => array(self::HAS_MANY, 'UbicacionTec', 'supervisor'),
			'ubicacionT0' => array(self::BELONGS_TO, 'UbicacionTec', 'ubicacionT'),
		);
	}

	public function behaviors()
	{
		return array('CAdvancedArBehavior',
				array('class' => 'ext.CAdvancedArBehavior')
				);
	}

	public function attributeLabels()
	{
		return array(
			'id' => Yii::t('app', 'ID'),
			'Username' => Yii::t('app', 'Username'),
			'Password' => Yii::t('app', 'Password'),
			'Analista' => Yii::t('app', 'Nombres'),
			'Proceso' => Yii::t('app', 'Proceso'),
			'Es_administrador' => Yii::t('app', 'Es Administrador'),
                        'Es_analista' => Yii::t('app', 'Es Ingeniero'),
                    
			
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Username',$this->Username,true);

		//$criteria->compare('password',$this->password,true);

		$criteria->compare('Analista',$this->Analista,true);

		$criteria->compare('Proceso',$this->Proceso,true);

		$criteria->compare('Es_administrador',$this->Es_administrador);
$criteria->compare('Es_analista',$this->Es_analista);
		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
