<?php

class UbicacionTec extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'ubicacionTec';
	}

	public function rules()
	{
		return array(
			array('supervisor', 'required'),
			array('supervisor', 'numerical', 'integerOnly'=>true),
			array('codigoSAP, padre', 'length', 'max'=>64),
			array('descripcion', 'length', 'max'=>128),
			array('id, codigoSAP, descripcion, padre, supervisor', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'metaDocs' => array(self::HAS_MANY, 'MetaDocs', 'ubicacionT'),
			'padre0' => array(self::BELONGS_TO, 'UbicacionTec', 'padre'),
			'ubicacionTecs' => array(self::HAS_MANY, 'UbicacionTec', 'padre'),
			'supervisor0' => array(self::BELONGS_TO, 'Usuarios', 'supervisor'),
			'usuarioses' => array(self::HAS_MANY, 'Usuarios', 'ubicacionT'),
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
			'codigoSAP' => Yii::t('app', 'Codigo Sap'),
			'descripcion' => Yii::t('app', 'Descripcion'),
			'padre' => Yii::t('app', 'Padre'),
			'supervisor' => Yii::t('app', 'Supervisor'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('codigoSAP',$this->codigoSAP,true);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('padre',$this->padre,true);

		$criteria->compare('supervisor',$this->supervisor);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
