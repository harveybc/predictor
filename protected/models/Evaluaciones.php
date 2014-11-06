<?php

class Evaluaciones extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'evaluaciones';
	}

	public function rules()
	{
		return array(
			array('usuario, fecha, evaluacionGeneral', 'required'),
			array('usuario, evaluacionGeneral', 'numerical', 'integerOnly'=>true),
			array('pregunta1, pregunta2, pregunta3, pregunta4, pregunta5, pregunta6, pregunta7, pregunta8, pregunta9, pregunta10', 'numerical'),
			array('id, usuario, fecha, evaluacionGeneral, pregunta1, pregunta2, pregunta3, pregunta4, pregunta5, pregunta6, pregunta7, pregunta8, pregunta9, pregunta10', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'usuario0' => array(self::BELONGS_TO, 'Usuarios', 'usuario'),
			'evaluacionGeneral0' => array(self::BELONGS_TO, 'EvaluacionesGenerales', 'evaluacionGeneral'),
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
			'usuario' => Yii::t('app', 'Usuario'),
			'fecha' => Yii::t('app', 'Fecha'),
			'evaluacionGeneral' => Yii::t('app', 'Evaluacion General'),
			'pregunta1' => Yii::t('app', 'Pregunta1'),
			'pregunta2' => Yii::t('app', 'Pregunta2'),
			'pregunta3' => Yii::t('app', 'Pregunta3'),
			'pregunta4' => Yii::t('app', 'Pregunta4'),
			'pregunta5' => Yii::t('app', 'Pregunta5'),
			'pregunta6' => Yii::t('app', 'Pregunta6'),
			'pregunta7' => Yii::t('app', 'Pregunta7'),
			'pregunta8' => Yii::t('app', 'Pregunta8'),
			'pregunta9' => Yii::t('app', 'Pregunta9'),
			'pregunta10' => Yii::t('app', 'Pregunta10'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('usuario',$this->usuario);

		$criteria->compare('fecha',$this->fecha,true);

		$criteria->compare('evaluacionGeneral',$this->evaluacionGeneral);

		$criteria->compare('pregunta1',$this->pregunta1);

		$criteria->compare('pregunta2',$this->pregunta2);

		$criteria->compare('pregunta3',$this->pregunta3);

		$criteria->compare('pregunta4',$this->pregunta4);

		$criteria->compare('pregunta5',$this->pregunta5);

		$criteria->compare('pregunta6',$this->pregunta6);

		$criteria->compare('pregunta7',$this->pregunta7);

		$criteria->compare('pregunta8',$this->pregunta8);

		$criteria->compare('pregunta9',$this->pregunta9);

		$criteria->compare('pregunta10',$this->pregunta10);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
