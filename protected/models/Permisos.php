<?php

class Permisos extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'permisos';
	}

	public function rules()
	{
		return array(
			array('modulo, usuario, operacion, documento', 'numerical', 'integerOnly'=>true),
			array('id, modulo, usuario, operacion, documento', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'modulo0' => array(self::BELONGS_TO, 'Modulos', 'modulo'),
			'usuario0' => array(self::BELONGS_TO, 'Usuarios', 'usuario'),
			'operacion0' => array(self::BELONGS_TO, 'Operaciones', 'operacion'),
			'documento0' => array(self::BELONGS_TO, 'Documentos', 'documento'),
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
			'modulo' => Yii::t('app', 'Modulo'),
			'usuario' => Yii::t('app', 'Usuario'),
			'operacion' => Yii::t('app', 'Operacion'),
			'documento' => Yii::t('app', 'Documento'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('modulo',$this->modulo);

		$criteria->compare('usuario',$this->usuario);

		$criteria->compare('operacion',$this->operacion);

		$criteria->compare('documento',$this->documento);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
